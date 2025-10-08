"""
Reconstrucción de potenciales escalares con:

Validación numérica estricta
Manejo seguro de hiperparámetros
Interpolación estable en todo el rango
Carga robusta con reconstrucción de nodos si faltan
"""
from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from typing import Iterable, Union, Optional, Tuple, Dict, Any
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.exceptions import NotFittedError
import logging
import dill as pickle  # Usar dill para mejor compatibilidad en serialización
logger = logging.getLogger('DVT_POTENTIAL')

def _as_1d_array(x: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr

class PotentialGP:
    """Proceso Gaussiano mejorado para reconstrucción de potenciales cosmológicos."""

    # Valores por defecto para reconstruir nodos si no vienen en el pickle
    _DEFAULT_ANCHORS: int = 201
    _DEFAULT_SPAN: Tuple[float, float] = (-1.5, 1.5)

    def __init__(
        self,
        nodes: Union[np.ndarray, Iterable[float], None] = None,
        logA: float = 0.0,
        logL: float = 0.0,
        alpha: float = 1e-12,
        n_anchors: Optional[int] = 21,
        anchor_span: Optional[Tuple[float, float]] = (-1.0, 1.0),
        optimizer: Optional[str] = "fmin_l_bfgs_b",
        normalize_y: bool = True,
        physical_scale: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inicializa el GP con validación exhaustiva de parámetros."""
        # Metadatos (p.ej. phi_span, n_anchors originales)
        self.meta: Dict[str, Any] = dict(meta or {})
        
        # Validación y configuración de nodos
        self._configure_nodes(nodes, n_anchors, anchor_span)
        
        # Configuración de hiperparámetros
        self.alpha = self._validate_alpha(alpha)
        self.optimizer = optimizer
        self.normalize_y = normalize_y
        self.V_scale = self._validate_scale(physical_scale)
        
        # Estado inicial
        self._last_theta: Optional[np.ndarray] = None
        self._fitted: bool = False
        
        # Inicialización del GP
        self._init_gp(*self._validate_hyperparams(logA, logL))
        
        logger.info(
            f"GP inicializado con {len(self.nodes)} nodos en "
            f"[{self.nodes.min():.3g}, {self.nodes.max():.3g}] | "
            f"Escala física: {self.V_scale:.3g}"
        )

    # ---------- Utilidades de estado ----------
    @property
    def nodes(self) -> np.ndarray:
        """Devuelve los nodos de anclaje como array 1D."""
        return self._nodes.flatten()

    def _default_nodes(self) -> np.ndarray:
        """Genera nodos por defecto usando meta si existe."""
        span = tuple(self.meta.get("phi_span", self._DEFAULT_SPAN))
        n = int(self.meta.get("n_anchors", self._DEFAULT_ANCHORS))
        lo, hi = map(float, span)
        
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = self._DEFAULT_SPAN
        if n < 2:
            n = self._DEFAULT_ANCHORS
            
        return np.linspace(lo, hi, n).reshape(-1, 1)

    # ---------- Configuración de nodos ----------
    def _configure_nodes(
        self,
        nodes: Union[np.ndarray, Iterable[float], None],
        n_anchors: Optional[int],
        anchor_span: Optional[Tuple[float, float]],
    ) -> None:
        """Configura los nodos con validación robusta."""
        if nodes is None:
            # Si no hay nodos explícitos, usa span + n_anchors si están dados, si no meta/default
            if n_anchors is not None and anchor_span is not None:
                lo, hi = map(float, anchor_span)
                if hi <= lo:
                    raise ValueError("El rango anchor_span debe ser (min, max)")
                n = int(n_anchors)
                if n < 2:
                    raise ValueError("n_anchors debe ser >= 2")
                self._nodes = np.linspace(lo, hi, n).reshape(-1, 1)
                # guarda en meta
                self.meta.setdefault("phi_span", (lo, hi))
                self.meta.setdefault("n_anchors", n)
            else:
                # Reconstrucción por defecto (útil en load() cuando no venían nodos)
                self._nodes = self._default_nodes()
        else:
            nodes_arr = _as_1d_array(nodes)
            if nodes_arr.size < 2:
                raise ValueError("Se requieren al menos 2 nodos en formato 1D")
            if not np.all(np.isfinite(nodes_arr)):
                raise ValueError("Nodos contienen valores no finitos")
            nodes_arr = np.unique(nodes_arr)
            self._nodes = nodes_arr.reshape(-1, 1)
            # Actualiza meta a partir de nodos explícitos
            self.meta["phi_span"] = (float(nodes_arr.min()), float(nodes_arr.max()))
            self.meta["n_anchors"] = int(nodes_arr.size)

    # ---------- Validaciones ----------
    def _validate_alpha(self, alpha: float) -> float:
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha debe ser positivo y finito")
        return alpha

    def _validate_scale(self, scale: float) -> float:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("La escala física debe ser positiva y finita")
        return scale

    def _validate_hyperparams(self, logA: float, logL: float) -> Tuple[float, float]:
        logA, logL = map(float, (logA, logL))
        if not (-5.0 <= logA <= 5.0) or not (-5.0 <= logL <= 5.0):
            raise ValueError("logA y logL deben estar en [-5, 5]")
        
        A, L = np.exp(logA), np.exp(logL)
        if not (np.isfinite(A) and np.isfinite(L)) or A <= 0 or L <= 0:
            raise ValueError("Hiperparámetros resultantes inválidos")
        return A, L

    # ---------- Inicialización y ajuste ----------
    def _init_gp(self, A: float, L: float) -> None:
        """Configura el GP con kernel RBF optimizable y ajusta anclas."""
        self._gp = GaussianProcessRegressor(
            kernel=C(A, (1e-6, 1e6)) * RBF(length_scale=L, length_scale_bounds=(1e-6, 1e6)),
            alpha=self.alpha,
            optimizer=self.optimizer,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=3,
            random_state=42,
        )
        self._fit_anchors()

    def _fit_anchors(self) -> None:
        """Ajusta el GP a nodos de anclaje (targets cero)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gp.fit(self._nodes, np.zeros_like(self._nodes))
                self._fitted = True
        except Exception as e:
            self._fitted = False
            logger.error(f"Error en ajuste inicial del GP: {str(e)}", exc_info=True)
            raise RuntimeError(f"Fallo en _fit_anchors: {str(e)}") from e

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            self._fit_anchors()

    # ---------- Entrada / salida ----------
    def _validate_input(self, phi: Union[float, np.ndarray]) -> np.ndarray:
        phi_arr = np.asarray(phi, dtype=float)
        if not np.all(np.isfinite(phi_arr)):
            raise ValueError("Input contiene valores no finitos")
        if np.any(np.abs(phi_arr) > 10.0):
            logger.warning("Valores de |φ| > 10 pueden requerir extrapolación")
        return phi_arr.reshape(-1, 1)

    def update(self, theta: Union[np.ndarray, Iterable[float]]) -> None:
        """Actualiza hiperparámetros solo si cambian significativamente.
        theta = [logA, logL]
        """
        theta_arr = np.asarray(theta, dtype=float).ravel()
        if theta_arr.size != 2:
            raise ValueError("theta debe contener exactamente [logA, logL]")
        
        if self._last_theta is None or not np.allclose(theta_arr, self._last_theta, rtol=1e-5):
            self._last_theta = theta_arr.copy()
            A, L = self._validate_hyperparams(*theta_arr)
            self._init_gp(A, L)
            logger.debug(f"GP actualizado: A={A:.3g}, L={L:.3g}")

    def set_scale(self, scale: float) -> None:
        self.V_scale = self._validate_scale(scale)
        logger.info(f"Escala física actualizada: {self.V_scale:.3g}")

    def set_nodes(self, nodes: Union[np.ndarray, Iterable[float]]) -> None:
        """Actualiza los nodos de anclaje con reajuste automático."""
        self._configure_nodes(nodes, n_anchors=None, anchor_span=None)
        self._fit_anchors()
        logger.info(f"Nodos actualizados a {len(self.nodes)} puntos")

    def set_alpha(self, alpha: float) -> None:
        self.alpha = self._validate_alpha(alpha)
        self._gp.alpha = self.alpha
        self._fit_anchors()
        logger.info(f"Parámetro alpha actualizado: {self.alpha:.3g}")

    def hyperparams(self) -> dict:
        """Devuelve hiperparámetros actuales como diccionario (seguro si no hay kernel)."""
        try:
            k = self._gp.kernel
            return {
                "A": float(k.k1.constant_value),
                "length_scale": float(k.k2.length_scale),
                "logA": float(np.log(k.k1.constant_value)),
                "logL": float(np.log(k.k2.length_scale)),
                "alpha": float(self._gp.alpha),
                "fitted": self._fitted,
            }
        except (NotFittedError, AttributeError):
            # Usa último theta si existe; de lo contrario NaN/alpha actual
            logA, logL = (self._last_theta if self._last_theta is not None else (np.nan, np.nan))
            return {
                "A": np.nan if np.isnan(logA) else float(np.exp(logA)),
                "length_scale": np.nan if np.isnan(logL) else float(np.exp(logL)),
                "logA": float(logA),
                "logL": float(logL),
                "alpha": float(self.alpha),
                "fitted": self._fitted,
            }

    # ---------- Evaluación ----------
    def V(
        self,
        phi: Union[float, np.ndarray],
        return_std: bool = False,
    ) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        """Evalúa el potencial normalizado V(φ)."""
        self._ensure_fitted()
        phi_arr = self._validate_input(phi)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if return_std:
                    mu, std = self._gp.predict(phi_arr, return_std=True)
                    return mu.squeeze(), std.squeeze()
                return self._gp.predict(phi_arr).squeeze()
        except Exception as e:
            logger.error(f"Error en V(φ): {str(e)}", exc_info=True)
            raise RuntimeError(f"Fallo en predicción: {str(e)}") from e

    def physical_potential(
        self,
        phi: Union[float, np.ndarray],
        return_std: bool = False,
    ) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        """Evalúa el potencial físico escalado V_phys = V_scale * V(φ)."""
        out = self.V(phi, return_std=return_std)
        if return_std:
            mu, std = out
            return mu * self.V_scale, std * self.V_scale
        return out * self.V_scale

    # ---------- Serialización ----------
    def save(self, path: Union[str, Path]) -> None:
        """Serializa el modelo a disco con manejo seguro de archivos."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            hp = self.hyperparams()
            
            # Intenta obtener logA/logL válidos para compatibilidad futura
            if np.isnan(hp.get("logA", np.nan)) or np.isnan(hp.get("logL", np.nan)):
                if self._last_theta is not None:
                    logA, logL = map(float, self._last_theta)
                else:
                    logA, logL = 0.0, 0.0
            else:
                logA, logL = float(hp["logA"]), float(hp["logL"])
                
            state = {
                "nodes": self._nodes,  # guardado como (N,1)
                "hyperparams": {"logA": logA, "logL": logL},
                "alpha": float(self.alpha),
                "V_scale": float(self.V_scale),
                "optimizer": self.optimizer,
                "normalize_y": bool(self.normalize_y),
                "fitted": bool(self._fitted),
                "meta": dict(self.meta),
                # Compat: guardar también 1D por si otro loader lo requiere
                "nodes_1d": self.nodes.copy(),
            }
            
            with path.open("wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.info(f"Modelo guardado en {path}")
            
        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}", exc_info=True)
            raise RuntimeError(f"Fallo en save(): {str(e)}") from e

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PotentialGP":
        """Carga un modelo PotentialGP desde un archivo pickle, con soporte para pickles de train_gp.py."""
        path = Path(path)
        try:
            with path.open("rb") as f:
                state = pickle.load(f)
                
            # Caso especial: Pickle de train_gp.py (dict con "gp" y "args")
            if isinstance(state, dict) and "gp" in state and "args" in state:
                # Importar BayesianEnsembleGP aquí para evitar dependencia circular
                try:
                    from genesis_modular.train_gp import BayesianEnsembleGP
                except ImportError:
                    raise ImportError("No se puede cargar pickle de train_gp.py sin la clase BayesianEnsembleGP")
                
                args = state.get("args", {})
                phi_min = args.get("phi_min", -1.5)  # Default si no está
                phi_max = args.get("phi_max", 1.5)
                n_points = int(args.get("n_points", cls._DEFAULT_ANCHORS))
                m_phi = args.get("m_phi", 0.5)  # Para escala física
                
                logger.warning("⚠️ Detectado pickle de train_gp.py (BayesianEnsembleGP); convirtiendo a PotentialGP")
                
                # Extraer hiperparámetros aproximados del ensamble
                try:
                    gp = state["gp"]
                    if hasattr(gp, "hyperparams") and callable(gp.hyperparams):
                        hp = gp.hyperparams()
                        logA = float(hp.get("logA", 0.0))
                        logL = float(hp.get("logL", 0.0))
                    else:
                        # Intentar aproximar desde modelos internos si existen
                        logA = 0.0
                        try:
                            length_scales = [
                                m["kernel"].k2.length_scale
                                for m in getattr(gp, "models", [])
                                if hasattr(m.get("kernel", None), "k2") and m.get("weight", 0) > 0
                            ]
                            logL = float(np.log(np.mean(length_scales))) if length_scales else 0.0
                        except Exception:
                            logL = 0.0
                except Exception as e:
                    logger.warning(f"No se pudieron extraer hiperparámetros: {e}. Usando defaults logA=0, logL=0")
                    logA, logL = 0.0, 0.0
                
                # Crear instancia de PotentialGP (el init ya configura y ajusta los anclajes)
                instance = cls(
                    nodes=np.linspace(phi_min, phi_max, n_points),
                    logA=logA,
                    logL=logL,
                    alpha=1e-12,  # Valor típico para estabilidad
                    n_anchors=n_points,
                    anchor_span=(phi_min, phi_max),
                    optimizer="fmin_l_bfgs_b",
                    normalize_y=True,
                    physical_scale=0.5 * (float(m_phi) ** 2) * (phi_max ** 2),  # Escala cosmológica
                    meta={"source": "train_gp_conversion", "original_args": args},
                )
                
                logger.info(f"Conversión completada: PotentialGP con rango [{phi_min}, {phi_max}], {n_points} nodos")
                return instance
                
            # Caso original: Pickle nativo de PotentialGP
            if not isinstance(state, dict):
                raise ValueError("Archivo pickle corrupto: no es un diccionario")
                
            # Recuperar nodos (soporta 'nodes' en (N,1) o 'nodes_1d')
            nodes = state.get("nodes", None)
            if nodes is None:
                nodes = state.get("nodes_1d", None)
            if nodes is None:
                raise ValueError("Archivo pickle incompleto: faltan 'nodes'")
                
            nodes_arr = np.asarray(nodes)
            
            # Recuperar hiperparámetros guardados
            hyperparams = state.get("hyperparams")
            if not isinstance(hyperparams, dict):
                raise ValueError("Archivo pickle incompleto: 'hyperparams' inválido o ausente")
                
            # Valores auxiliares
            alpha = float(state.get("alpha", 1e-8))
            n_anchors = int(state.get("n_anchors", nodes_arr.shape[0] if nodes_arr is not None else cls._DEFAULT_ANCHORS))
            anchor_span = state.get("anchor_span", tuple(state.get("meta", {}).get("phi_span", cls._DEFAULT_SPAN)))
            optimizer = state.get("optimizer", "fmin_l_bfgs_b")
            normalize_y = bool(state.get("normalize_y", True))
            V_scale = float(state.get("V_scale", state.get("physical_scale", 1.0)))
            meta = dict(state.get("meta", {}))
            
            # Validaciones
            try:
                logA = float(hyperparams.get("logA", 0.0))
                logL = float(hyperparams.get("logL", 0.0))
            except Exception:
                raise ValueError("Hiperparámetros inválidos en pickle")
                
            if not np.all(np.isfinite(nodes_arr.astype(float))):
                raise ValueError("Nodos contienen valores no finitos")
                
            # Crear instancia; init valida y ajusta
            instance = cls(
                nodes=nodes_arr,
                logA=logA,
                logL=logL,
                alpha=alpha,
                n_anchors=n_anchors,
                anchor_span=tuple(anchor_span),
                optimizer=optimizer,
                normalize_y=normalize_y,
                physical_scale=V_scale,
                meta=meta,
            )
            
            # Restaurar estado de ajuste si se guardó
            instance._fitted = bool(state.get("fitted", instance._fitted))
            # si se marcó como ajustado pero init no lo hizo por alguna razón, intentar ajustar anclas
            if instance._fitted and not getattr(instance, "_fitted", False):
                instance._fit_anchors()
                
            return instance
            
        except Exception as e:
            logger.error(f"Error cargando modelo GP: {str(e)}", exc_info=True)
            raise RuntimeError(f"Fallo en load(): {str(e)}") from e

    # ---------- Representación ----------
    def __repr__(self) -> str:
        params = self.hyperparams()
        A = params.get("A", np.nan)
        L = params.get("length_scale", np.nan)
        return (
            f"PotentialGP(A={A:.3g}, L={L:.3g}, "
            f"fitted={params.get('fitted', False)}, scale={self.V_scale:.3g})"
        )

    __call__ = V  # Sintaxis alternativa: gp(phi) -> V(phi)

__all__ = ["PotentialGP"] # ojalá funcione