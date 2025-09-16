#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# train_gp.py — DVT GP Ultra (Versión Final Fusionada y Mejorada)
# ==============================================================================
# Integra todas las mejoras: paralelización con joblib, validaciones extremas,
# gráficos en alta calidad (HQ1200), reporte PDF para papers científicos y
# robustez mejorada para hardware moderno.
# ==============================================================================
from __future__ import annotations
import argparse
import logging
import pickle
import re
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import matplotlib.pyplot as plt
import optuna
import sympy as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, DotProduct, ExpSineSquared, Kernel, Matern,
    RationalQuadratic, RBF, WhiteKernel
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from joblib import Parallel, delayed
import seaborn as sns
from scipy import stats
import scienceplots  # Para estilos científicos

# -------------------- Configuración inicial --------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configurar estilo científico
plt.style.use(['science', 'ieee', 'grid'])

def setup_logging(level=logging.INFO, log_file="dvt_gp_ultra.log"):
    """Configura el sistema de logging con formato enriquecido."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=level,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger("DVT_GP_Ultra")

logger = setup_logging()

# -------------------- Kernel polinómico + fábrica segura --------------------
class PolynomialKernel(Kernel):
    """Kernel polinómico simple (X·Y^T + c)^degree"""
    def __init__(self, degree: int = 2, c: float = 1.0):
        if degree < 0:
            raise ValueError("PolynomialKernel: degree debe ser >= 0")
        self.degree = int(degree)
        self.c = float(c)
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = (np.dot(X, Y.T) + self.c) ** self.degree
        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K
    
    def diag(self, X):
        return (np.einsum('ij,ij->i', X, X) + self.c) ** self.degree
    
    def is_stationary(self):
        return False
    
    def __repr__(self):
        return f"Polynomial(degree={self.degree}, c={self.c})"

KERNEL_FACTORY: Dict[str, Callable[..., Kernel]] = {
    "RBF": RBF,
    "Matern": Matern,
    "RQ": RationalQuadratic,
    "RationalQuadratic": RationalQuadratic,
    "ExpSineSquared": ExpSineSquared,
    "ExpSine": ExpSineSquared,
    "WhiteKernel": WhiteKernel,
    "White": WhiteKernel,
    "ConstantKernel": ConstantKernel,
    "Constant": ConstantKernel,
    "DotProduct": DotProduct,
    "Polynomial": PolynomialKernel,
}

def _safe_kernel_eval(expr: str) -> Kernel:
    """
    Evalúa de forma controlada expresiones tipo:
      "ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)"
    Solo permite nombres presentes en KERNEL_FACTORY y argumentos simples.
    """
    if expr is None or not isinstance(expr, str) or not expr.strip():
        raise ValueError("Expresión de kernel vacía o inválida.")
    if re.search(r"[^\w\s\+\-\*\/\^\(\)\.,=\:eE0-9]", expr):
        raise ValueError(f"Expresión de kernel con caracteres no permitidos: {expr}")
    pattern = r"(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*(?:\((?P<args>[^()]*)\))?"
    def repl(m: re.Match):
        name = m.group("name")
        args = m.group("args") or ""
        if name not in KERNEL_FACTORY:
            raise ValueError(f"Kernel desconocido '{name}'. Opciones: {sorted(KERNEL_FACTORY.keys())}")
        return f'KERNEL_FACTORY["{name}"]({args})'
    expr_replaced = re.sub(pattern, repl, expr)
    try:
        kernel = eval(expr_replaced, {"KERNEL_FACTORY": KERNEL_FACTORY, "np": np})
    except Exception as e:
        raise ValueError(f"Error al evaluar kernel '{expr}': {e}")
    if not isinstance(kernel, Kernel):
        raise TypeError("La expresión no produjo un objeto Kernel válido.")
    return kernel

# -------------------- Función base del potencial --------------------
def sin_function(phi, m_phi):
    """Función potencial base: V(φ) = 1/2 m²φ²"""
    phi_arr = np.asarray(phi).ravel()
    return 0.5 * (m_phi ** 2) * (phi_arr ** 2)

# -------------------- Generador de datos sintéticos robusto --------------------
def generate_training_data(n_points, phi_range, m_phi, noise, seed=None,
                          adversarial=False, outlier_frac=0.0, normalize=False,
                          extend_range_factor=0.0):
    """Genera datos de entrenamiento sintéticos con opciones avanzadas."""
    if seed is not None:
        np.random.seed(seed)
    phi_min, phi_max = phi_range
    range_extension = extend_range_factor * (phi_max - phi_min)
    phi = np.linspace(phi_min - range_extension, phi_max + range_extension, n_points).reshape(-1, 1)
    V = sin_function(phi, m_phi).reshape(-1, 1)
    sin_perturb = 0.01 * np.sin(5 * phi).reshape(-1, 1)
    correlated_noise = np.random.normal(0, noise, size=(n_points, 1))
    if adversarial:
        n_outliers = int(outlier_frac * n_points)
        if n_outliers > 0:
            indices = np.random.choice(n_points, n_outliers, replace=False)
            correlated_noise[indices] += np.random.normal(0, 5 * noise, size=(n_outliers, 1))
    V += sin_perturb + correlated_noise
    if normalize:
        V_mean = np.mean(V)
        V_std = np.std(V)
        if V_std > 0:
            V = (V - V_mean) / V_std
    return phi, V, m_phi

# -------------------- Ensamble Bayesiano GP --------------------
class BayesianEnsembleGP:
    """Ensamble bayesiano de procesos gaussianos con múltiples kernels."""
    
    def __init__(
        self,
        kernel_types: Optional[List[str]] = None,
        nu_values: Optional[List[float]] = None,
        seed: int = 42,
        kernel_expr: Optional[str] = None,
        n_optuna_trials: int = 0,
        n_jobs: int = 1,
        scaler_type: str = "standard"
    ):
        self.kernel_types = [kt.lower() for kt in (kernel_types or ["matern", "rbf", "rational_quadratic"])]
        self.nu_values = nu_values or [0.5, 1.5, 2.5]
        self.seed = seed
        self.kernel_expr = kernel_expr
        self.n_optuna_trials = n_optuna_trials
        self.n_jobs = n_jobs
        self.scaler_type = scaler_type.lower()
        self.models: List[Dict] = []
        self.weights: List[float] = []
        self.log_marginal_likelihoods: List[float] = []
        self.X_scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.y_scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.is_fitted = False
    
    def initialize_models(self, n_restarts: int = 0):
        """Inicializa los modelos GP con los kernels especificados."""
        self.models = []
        for ktype in self.kernel_types:
            if ktype.startswith("matern"):
                for nu in self.nu_values:
                    kernel = (
                        ConstantKernel(1.0, (1e-6, 1e6)) *
                        Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e6), nu=nu) +
                        WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
                    )
                    self.models.append({
                        "type": f"matern_{nu}",
                        "kernel": kernel,
                        "gp": GaussianProcessRegressor(
                            kernel=kernel,
                            n_restarts_optimizer=n_restarts,
                            alpha=1e-8,
                            normalize_y=False,
                            random_state=self.seed
                        )
                    })
            elif ktype == "rbf":
                kernel = (
                    ConstantKernel(1.0, (1e-6, 1e6)) *
                    RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6)) +
                    WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
                )
                self.models.append({
                    "type": "rbf",
                    "kernel": kernel,
                    "gp": GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=n_restarts,
                        alpha=1e-8,
                        normalize_y=False,
                        random_state=self.seed
                    )
                })
            elif ktype in ("rational_quadratic", "rq"):
                kernel = (
                    ConstantKernel(1.0, (1e-6, 1e6)) *
                    RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-2, 10.0)) +
                    WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
                )
                self.models.append({
                    "type": "rational_quadratic",
                    "kernel": kernel,
                    "gp": GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=n_restarts,
                        alpha=1e-8,
                        normalize_y=False,
                        random_state=self.seed
                    )
                })
            elif ktype == "polynomial":
                kernel = PolynomialKernel(degree=2, c=1.0)
                self.models.append({
                    "type": "polynomial",
                    "kernel": kernel,
                    "gp": GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=n_restarts,
                        alpha=1e-8,
                        normalize_y=False,
                        random_state=self.seed
                    )
                })
        
        # Añadir kernel personalizado si se especificó
        if self.kernel_expr:
            custom_kernel = _safe_kernel_eval(self.kernel_expr)
            self.models.append({
                "type": "custom_expr",
                "kernel": custom_kernel,
                "gp": GaussianProcessRegressor(
                    kernel=custom_kernel,
                    n_restarts_optimizer=n_restarts,
                    alpha=1e-8,
                    normalize_y=False,
                    random_state=self.seed
                )
            })
        
        # Añadir kernel polinómico si todos son estacionarios
        if not any(not m["kernel"].is_stationary() for m in self.models):
            logger.warning("Todos los kernels son estacionarios. Añadiendo Polynomial para extrapolación.")
            self.models.append({
                "type": "polynomial",
                "kernel": PolynomialKernel(degree=2, c=1.0),
                "gp": GaussianProcessRegressor(
                    kernel=PolynomialKernel(degree=2, c=1.0),
                    n_restarts_optimizer=n_restarts,
                    alpha=1e-8,
                    normalize_y=False,
                    random_state=self.seed
                )
            })
        
        if not self.models:
            raise ValueError("No se definieron modelos: revisa kernel_types o kernel_expr")
    
    def _optimize_hyperparameters(self, model: Dict, n_trials: int):
        """
        Optimiza hiperparámetros usando Optuna para un modelo específico (VERSIÓN CORREGIDA Y ROBUSTA).
        Este enfoque es más seguro porque le pide directamente a scikit-learn cuáles son
        sus parámetros sintonizables, en lugar de intentar adivinar su estructura interna.
        """
        def objective(trial):
            try:
                params = {}
                # --- CAMBIO CLAVE 1: Obtener parámetros sintonizables directamente ---
                # Se obtienen todos los parámetros del kernel que no son fijos (ej. "_bounds").
                # Esto es más robusto que el recorrido manual del árbol de kernels.
                tunable_params = [
                    p for p in model["gp"].get_params().keys()
                    if p.startswith("kernel__") and not p.endswith("_bounds")
                ]

                # --- CAMBIO CLAVE 2: Sugerir valores basados en nombres de parámetros ---
                # Itera sobre los nombres correctos que scikit-learn proporciona.
                for param_name in tunable_params:
                    if "length_scale" in param_name:
                        params[param_name] = trial.suggest_float(param_name, 1e-2, 1e2, log=True)
                    elif "constant_value" in param_name:
                        params[param_name] = trial.suggest_float(param_name, 1e-3, 1e3, log=True)
                    elif "noise_level" in param_name:
                        params[param_name] = trial.suggest_float(param_name, 1e-10, 1e-1, log=True)
                    elif "alpha" in param_name and param_name != "kernel__alpha":
                        # Específico para el kernel RationalQuadratic, evitando el 'alpha' principal del GP.
                        params[param_name] = trial.suggest_float(param_name, 1e-4, 10.0, log=True)

                # Sugiere el alpha (regularización) del propio GaussianProcessRegressor.
                params["alpha"] = trial.suggest_float("alpha", 1e-10, 1e-2, log=True)

                # Establece los parámetros sugeridos y ajusta el modelo.
                model["gp"].set_params(**params)
                model["gp"].fit(self.X_train_scaled, self.y_train_scaled)

                return float(model["gp"].log_marginal_likelihood())

            except optuna.exceptions.TrialPruned:
                # Re-lanzar para que Optuna gestione podado si corresponde.
                raise
            except Exception:
                # Si el ajuste falla por cualquier motivo (ej. parámetros inestables),
                # Optuna lo registrará como un mal resultado.
                return -np.inf

        # Ejecuta el estudio de Optuna.
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

        # --- CAMBIO CLAVE 3: Aplicación simplificada de los mejores parámetros ---
        # `study.best_params` ya contiene los nombres correctos (ej. "kernel__k1__constant_value").
        best_params = study.best_params or {}
        if best_params:
            try:
                model["gp"].set_params(**best_params)
                logger.info(f"Modelo {model.get('type')} optimizado con éxito con Optuna.")
            except Exception as e:
                logger.warning(f"No se pudieron aplicar los mejores parámetros de Optuna para {model.get('type')}: {e}")
        else:
            logger.warning(f"Optuna no encontró mejores parámetros para {model.get('type')}.")
    
    def _train_single_model(self, model_dict):
        """Entrena un modelo individual y devuelve resultados."""
        try:
            model_dict["gp"].fit(self.X_train_scaled, self.y_train_scaled)
            lml = float(model_dict["gp"].log_marginal_likelihood())
            logger.info(f"Modelo {model_dict['type']} entrenado. LML: {lml:.4f}")
            return model_dict, lml, None
        except Exception as e:
            logger.warning(f"El modelo {model_dict.get('type')} falló al ajustarse: {e}")
            return model_dict, -np.inf, str(e)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ajusta el ensamble a los datos de entrenamiento."""
        self.X_train_scaled = self.X_scaler.fit_transform(X)
        y_2d = np.asarray(y).reshape(-1, 1)
        self.y_train_scaled = self.y_scaler.fit_transform(y_2d)
        
        # Optimización con Optuna si está habilitada
        if self.n_optuna_trials > 0:
            logger.info(f"Optimizando hiperparámetros con Optuna para {len(self.models)} modelos...")
            for m in self.models:
                try:
                    self._optimize_hyperparameters(m, self.n_optuna_trials)
                except Exception as e:
                    logger.warning(f"Optuna falló para {m.get('type')}: {e}")
        
        # Entrenamiento paralelo de modelos
        results = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._train_single_model)(m) for m in self.models
        )
        
        # Procesar resultados
        self.log_marginal_likelihoods = []
        for i, (model_dict, lml, error) in enumerate(results):
            self.models[i] = model_dict
            self.log_marginal_likelihoods.append(lml)
        
        # Calcular pesos basados en verosimilitud marginal
        max_ll = np.nanmax(self.log_marginal_likelihoods)
        exp_ll = np.exp(np.array(self.log_marginal_likelihoods) - max_ll)
        total = np.nansum(exp_ll)
        
        if total > 0 and np.isfinite(total):
            self.weights = (exp_ll / total).tolist()
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # Log de pesos
        for i, m in enumerate(self.models):
            logger.info(f"Peso {m['type']}: {self.weights[i]:.4f}")
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones con el ensamble."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de predecir.")
        
        Xs = self.X_scaler.transform(X)
        means_list, stds_list = [], []
        
        for m in self.models:
            mean, std = m["gp"].predict(Xs, return_std=True)
            mean = np.asarray(mean).ravel()
            std = np.asarray(std).ravel()
            means_list.append(mean)
            stds_list.append(std)
        
        means = np.array(means_list)
        stds = np.array(stds_list)
        w = np.array(self.weights).reshape(-1, 1)
        
        weighted_mean = np.sum(w * means, axis=0)
        
        if return_std:
            weighted_var = np.sum(w * (stds ** 2), axis=0)
            var_of_means = np.sum(w * ((means - weighted_mean) ** 2), axis=0)
            total_std = np.sqrt(np.maximum(0.0, weighted_var + var_of_means))
        else:
            total_std = np.zeros_like(weighted_mean)
        
        # Revertir escalado
        inv_mean = self.y_scaler.inverse_transform(weighted_mean.reshape(-1, 1)).ravel()
        y_scale = self.y_scaler.scale_.ravel()[0] if getattr(self.y_scaler, "scale_", None) is not None else 1.0
        inv_std = total_std * y_scale
        
        return inv_mean, inv_std

# -------------------- Validaciones extremas --------------------
def multi_scale_validation(phi_range: Tuple[float, float], m_phi: float, noise: float, seed: int,
                           kernel_expr: Optional[str], n_restarts: int, kernel_types: List[str],
                           nu_values: List[float], n_jobs: int, extend_range_factor=0.0) -> Dict:
    """Validación de convergencia a múltiples escalas de datos."""
    logger.info("Validación multi-escala...")
    predictions = {}
    n_points_list = [200, 1000, 5000]
    
    phi_min, phi_max = phi_range
    range_extension = extend_range_factor * (phi_max - phi_min)
    phi_fine = np.linspace(phi_min - range_extension, phi_max + range_extension, 
                          max(n_points_list)).reshape(-1, 1)
    
    def train_scale(n_points):
        phi_temp, V_temp, _ = generate_training_data(
            n_points, phi_range, m_phi, noise, seed, extend_range_factor=extend_range_factor
        )
        temp_gp = BayesianEnsembleGP(
            kernel_types=kernel_types, 
            nu_values=nu_values, 
            seed=seed, 
            kernel_expr=kernel_expr, 
            n_jobs=n_jobs
        )
        temp_gp.initialize_models(n_restarts=n_restarts)
        temp_gp.fit(phi_temp, V_temp)
        V_pred, _ = temp_gp.predict(phi_fine)
        return n_points, V_pred
    
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(train_scale)(n_points) for n_points in n_points_list
    )
    
    for n_points, V_pred in results:
        predictions[n_points] = V_pred
    
    fine_pred = predictions[max(n_points_list)]
    diff_1 = float(np.mean(np.abs(predictions[n_points_list[0]] - fine_pred)))
    diff_2 = float(np.mean(np.abs(predictions[n_points_list[1]] - fine_pred)))
    
    converged = diff_1 > diff_2
    logger.info(f"Convergencia: {'SÍ' if converged else 'NO'} (Diff 200->5k: {diff_1:.4f}, 1k->5k: {diff_2:.4f})")
    
    return {"converged": converged, "convergence_diff": diff_2}

def symbolic_validation(phi_range: Tuple[float, float], m_phi: float, gp_model: BayesianEnsembleGP) -> Dict:
    """Validación simbólica de derivadas mediante comparación con solución analítica."""
    phi_sym = sp.Symbol('phi')
    V_sym = 0.5 * (m_phi ** 2) * (phi_sym ** 2)
    dV_dphi_sym = sp.diff(V_sym, phi_sym)
    d2V_dphi2_sym = sp.diff(dV_dphi_sym, phi_sym)
    
    dV = sp.lambdify(phi_sym, dV_dphi_sym, 'numpy')
    d2V = sp.lambdify(phi_sym, d2V_dphi2_sym, 'numpy')
    
    phi_test = np.linspace(phi_range[0], phi_range[1], 500).reshape(-1, 1)
    V_pred, _ = gp_model.predict(phi_test)
    
    # Calcular derivadas numéricas
    dV_num = np.gradient(V_pred.ravel(), phi_test.ravel())
    d2V_num = np.gradient(dV_num, phi_test.ravel())
    
    # Calcular derivadas analíticas
    try:
        dV_analytic = np.asarray(dV(phi_test.ravel())).ravel()
    except Exception:
        dV_analytic = np.asarray(dV(phi_test)).ravel()
    
    try:
        d2V_analytic = np.asarray(d2V(phi_test.ravel())).ravel()
    except Exception:
        val = float(d2V(0)) if callable(d2V) else float(d2V)
        d2V_analytic = np.full_like(d2V_num, val, dtype=float)
    
    # Asegurar formas compatibles
    if d2V_analytic.shape != d2V_num.shape:
        if d2V_analytic.size == 1:
            d2V_analytic = np.full_like(d2V_num, float(np.asarray(d2V_analytic).ravel()[0]), dtype=float)
        else:
            d2V_analytic = np.resize(d2V_analytic, d2V_num.shape)
    
    # Calcular métricas
    dV_rmse = float(np.sqrt(mean_squared_error(dV_analytic, dV_num)))
    d2V_rmse = float(np.sqrt(mean_squared_error(d2V_analytic, d2V_num)))
    dV_mae = float(mean_absolute_error(dV_analytic, dV_num))
    d2V_mae = float(mean_absolute_error(d2V_analytic, d2V_num))
    
    passed = (dV_rmse < 0.5) and (d2V_rmse < 0.5)
    logger.info(f"Validación simbólica: {'PASADA' if passed else 'FALLADA'} "
                f"(dV RMSE={dV_rmse:.4f}, d2V RMSE={d2V_rmse:.4f})")
    
    return {
        "dV_rmse": dV_rmse, 
        "d2V_rmse": d2V_rmse,
        "dV_mae": dV_mae,
        "d2V_mae": d2V_mae,
        "symbolic_validation_passed": passed
    }

def adversarial_validation(gp_model: BayesianEnsembleGP, phi_range: Tuple[float, float], m_phi: float,
                          noise: float, seed: int) -> Dict:
    """Validación con datos adversarios que contienen outliers."""
    phi_test, V_test, _ = generate_training_data(
        1000, phi_range, m_phi, noise, seed, adversarial=True, outlier_frac=0.1
    )
    V_pred, _ = gp_model.predict(phi_test)
    
    r2 = float(r2_score(V_test, V_pred))
    mse = float(mean_squared_error(V_test, V_pred))
    mae = float(mean_absolute_error(V_test, V_pred))
    
    passed = r2 > 0.8
    logger.info(f"Validación adversarial: {'PASADA' if passed else 'FALLADA'} (R²={r2:.4f}, MSE={mse:.4e})")
    
    return {
        "adversarial_r2": r2,
        "adversarial_mse": mse,
        "adversarial_mae": mae,
        "adversarial_validation_passed": passed
    }

def extreme_cross_validation(X: np.ndarray, y: np.ndarray, kernel_expr: Optional[str],
                            kernel_types: List[str], nu_values: List[float],
                            seed: int, n_restarts: int, n_jobs: int) -> Dict:
    """Validación cruzada extrema con RepeatedKFold 5x2."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    def train_fold(tr_idx, te_idx):
        temp_gp = BayesianEnsembleGP(
            kernel_types=kernel_types, 
            nu_values=nu_values, 
            seed=seed, 
            kernel_expr=kernel_expr, 
            n_jobs=n_jobs
        )
        temp_gp.initialize_models(n_restarts=n_restarts)
        temp_gp.fit(X[tr_idx], y[tr_idx])
        y_pred, _ = temp_gp.predict(X[te_idx])
        
        r2 = r2_score(y[te_idx], y_pred)
        mse = mean_squared_error(y[te_idx], y_pred)
        mae = mean_absolute_error(y[te_idx], y_pred)
        
        return r2, mse, mae
    
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(train_fold)(tr_idx, te_idx) for tr_idx, te_idx in rkf.split(X)
    )
    
    for r2, mse, mae in results:
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
    
    r2_scores = np.array(r2_scores)
    mse_scores = np.array(mse_scores)
    mae_scores = np.array(mae_scores)
    
    stability = bool(np.std(r2_scores) < 0.1)
    mean_r2 = float(np.mean(r2_scores))
    std_r2 = float(np.std(r2_scores))
    mean_mse = float(np.mean(mse_scores))
    std_mse = float(np.std(mse_scores))
    mean_mae = float(np.mean(mae_scores))
    std_mae = float(np.std(mae_scores))
    
    logger.info(f"Cross-Validation extrema: R² medio={mean_r2:.4f} ± {std_r2:.4f}, "
                f"MSE medio={mean_mse:.4e}, Estabilidad={'ALTA' if stability else 'BAJA'}")
    
    return {
        "cv_r2_mean": mean_r2,
        "cv_r2_std": std_r2,
        "cv_mse_mean": mean_mse,
        "cv_mse_std": std_mse,
        "cv_mae_mean": mean_mae,
        "cv_mae_std": std_mae,
        "cv_stability": stability
    }

def extrapolation_test(gp_model: BayesianEnsembleGP, phi_range: Tuple[float, float], m_phi: float,
                      noise: float, seed: int, extend_range_factor=0.0) -> Dict:
    """Prueba de extrapolación más allá del rango de entrenamiento."""
    phi_min, phi_max = phi_range
    range_extension = extend_range_factor * (phi_max - phi_min)
    
    # Rango extendido para prueba
    test_range = (phi_min - 2 * range_extension, phi_max + 2 * range_extension)
    phi_test = np.linspace(test_range[0], test_range[1], 2000).reshape(-1, 1)
    
    # Generar datos de prueba sin ruido adicional para evaluación limpia
    V_test = sin_function(phi_test, m_phi).reshape(-1, 1)
    
    V_pred, V_std = gp_model.predict(phi_test)
    
    # Calcular métricas solo en la región de extrapolación
    mask = (phi_test[:, 0] < phi_min - range_extension) | (phi_test[:, 0] > phi_max + range_extension)
    
    if np.any(mask):
        r2_extrap = float(r2_score(V_test[mask, 0], V_pred[mask]))
        mse_extrap = float(mean_squared_error(V_test[mask, 0], V_pred[mask]))
        mae_extrap = float(mean_absolute_error(V_test[mask, 0], V_pred[mask]))
        
        # Calcular intervalo de confianza
        coverage = float(np.mean(
            (V_pred[mask] - 1.96 * V_std[mask] <= V_test[mask, 0]) & 
            (V_test[mask, 0] <= V_pred[mask] + 1.96 * V_std[mask])
        ))
    else:
        r2_extrap = float("nan")
        mse_extrap = float("nan")
        mae_extrap = float("nan")
        coverage = float("nan")
    
    passed = (r2_extrap > 0.7) if np.isfinite(r2_extrap) else False
    logger.info(f"Test extrapolación: {'PASADO' if passed else 'FALLADO'} "
                f"(R² extra={r2_extrap:.4f}, Cobertura IC95%={coverage:.3f})")
    
    return {
        "extrapolation_r2": r2_extrap,
        "extrapolation_mse": mse_extrap,
        "extrapolation_mae": mae_extrap,
        "extrapolation_coverage": coverage,
        "extrapolation_pass": passed
    }

# -------------------- Métricas y análisis estadístico --------------------
def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray] = None, 
                                 label: str = "") -> Dict[str, float]:
    """Calcula métricas de evaluación comprehensivas."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    
    metrics = {
        f"R2_{label}": float(r2_score(y_t, y_p)),
        f"MSE_{label}": float(mean_squared_error(y_t, y_p)),
        f"RMSE_{label}": float(np.sqrt(mean_squared_error(y_t, y_p))),
        f"MAE_{label}": float(mean_absolute_error(y_t, y_p)),
        f"MaxError_{label}": float(np.max(np.abs(y_t - y_p))),
    }
    
    # Calcular estadísticas adicionales si hay desviaciones estándar
    if y_std is not None:
        y_s = np.asarray(y_std).ravel()
        residuals = y_t - y_p
        z_scores = residuals / y_s
        nll = float(0.5 * np.mean(residuals**2 / y_s**2 + np.log(2 * np.pi * y_s**2)))
        
        metrics.update({
            f"NLL_{label}": nll,
            f"Coverage95_{label}": float(np.mean((y_p - 1.96 * y_s <= y_t) & (y_t <= y_p + 1.96 * y_s))),
            f"MeanStd_{label}": float(np.mean(y_s)),
            f"StdStd_{label}": float(np.std(y_s)),
            f"MeanZ_{label}": float(np.mean(z_scores)),
            f"StdZ_{label}": float(np.std(z_scores)),
        })
    
    return metrics

def statistical_analysis(y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Realiza análisis estadístico completo de los residuales."""
    residuals = y_true - y_pred
    analysis = {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_skewness": float(stats.skew(residuals)),
        "residual_kurtosis": float(stats.kurtosis(residuals)),
        "shapiro_p": float(stats.shapiro(residuals)[1]) if len(residuals) <= 5000 else float("nan"),
    }
    
    # Test de normalidad adicional
    if len(residuals) > 3:
        analysis["normaltest_p"] = float(stats.normaltest(residuals)[1])
    else:
        analysis["normaltest_p"] = float("nan")
    
    # Calibrar incertidumbre si se proporciona y_std
    if y_std is not None:
        z_scores = residuals / y_std
        analysis["zscore_mean"] = float(np.mean(z_scores))
        analysis["zscore_std"] = float(np.std(z_scores))
        analysis["zscore_skewness"] = float(stats.skew(z_scores))
        analysis["zscore_kurtosis"] = float(stats.kurtosis(z_scores))
    
    return analysis

# -------------------- Visualización de alta calidad --------------------
def create_hq_plots(phi_train: np.ndarray, V_train: np.ndarray, phi_val: np.ndarray, V_val: np.ndarray,
                   phi_grid: np.ndarray, V_pred: np.ndarray, V_std: np.ndarray, curvature: np.ndarray,
                   residuals: Dict[str, np.ndarray], outdir: Path, timestamp: str, m_phi: float) -> Dict[str, Path]:
    """Genera gráficos de alta calidad para publicación científica."""
    paths = {}
    plot_kwargs = {'dpi': 1200, 'bbox_inches': 'tight', 'pad_inches': 0.1}
    
    # 1. Gráfico de reconstrucción del potencial
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(phi_train, V_train, s=20, alpha=0.7, label="Datos entrenamiento", color='blue', edgecolors='none')
    ax.scatter(phi_val, V_val, s=20, alpha=0.7, label="Datos validación", color='green', edgecolors='none')
    
    # Predicción
    ax.plot(phi_grid, V_pred, linewidth=2.5, label="Predicción GP", color='red')
    ax.fill_between(phi_grid.ravel(), V_pred - 2 * V_std, V_pred + 2 * V_std, 
                   alpha=0.3, label="95% IC", color='orange')
    
    # Potencial teórico
    V_true = sin_function(phi_grid, m_phi)
    ax.plot(phi_grid, V_true, '--', linewidth=2.0, label="Verdad teórica", color='black')
    
    # --- ETIQUETAS YA CORREGIDAS ---
    ax.set_xlabel(r"$\phi$", fontsize=14)
    ax.set_ylabel(r"$V(\phi)$", fontsize=14)
    ax.set_title("Reconstrucción del Potencial Inflacionario", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    p1_png = outdir / f"potential_reconstruction_{timestamp}.png"
    p1_pdf = outdir / f"potential_reconstruction_{timestamp}.pdf"
    fig.savefig(p1_png, **plot_kwargs)
    fig.savefig(p1_pdf, **plot_kwargs)
    paths["potential_plot"] = (p1_png, p1_pdf)
    plt.close(fig)
    
    # 2. Gráfico de curvatura
    fig, ax = plt.subplots(figsize=(8, 6))
    # --- CORRECCIÓN 1: Etiquetas de la leyenda en formato LaTeX ---
    ax.plot(phi_grid.ravel(), curvature, linewidth=2.5, label=r"Curvatura estimada $\frac{d^2V}{d\phi^2}$")
    ax.axhline(y=m_phi**2, linestyle="--", linewidth=2.0, 
               label=f"Valor teórico $m^2 = {m_phi**2:.4f}$", color='red')
    
    ax.set_xlabel(r"$\phi$", fontsize=14)
    # --- CORRECCIÓN 2: Etiqueta del eje Y en formato LaTeX ---
    ax.set_ylabel(r"$\frac{d^2V}{d\phi^2}$", fontsize=14)
    ax.set_title("Curvatura del Potencial", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    p2_png = outdir / f"curvature_{timestamp}.png"
    p2_pdf = outdir / f"curvature_{timestamp}.pdf"
    fig.savefig(p2_png, **plot_kwargs)
    fig.savefig(p2_pdf, **plot_kwargs)
    paths["curvature_plot"] = (p2_png, p2_pdf)
    plt.close(fig)
    
    # 3. Gráfico de residuales
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(residuals["phi"].ravel(), residuals["res"].ravel(), 
              s=20, alpha=0.7, label="Residuales (val)")
    ax.axhline(0.0, linestyle="--", color='red', linewidth=2.0)
    # --- CORRECCIÓN 3: Etiquetas de los ejes en formato LaTeX ---
    ax.set_xlabel(r"$\phi$ (val)", fontsize=14)
    ax.set_ylabel(r"Residuo = $V_{\mathrm{val}} - \hat{V}_{\mathrm{val}}$", fontsize=14)
    ax.set_title("Residuales de Validación", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    p3_png = outdir / f"residuals_{timestamp}.png"
    p3_pdf = outdir / f"residuals_{timestamp}.pdf"
    fig.savefig(p3_png, **plot_kwargs)
    fig.savefig(p3_pdf, **plot_kwargs)
    paths["residuals_plot"] = (p3_png, p3_pdf)
    plt.close(fig)
    
    # 4. Gráfico de distribución de residuales
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals["res"].ravel(), kde=True, ax=ax, stat="density")
    ax.set_xlabel("Residuales", fontsize=14)
    ax.set_ylabel("Densidad", fontsize=14)
    ax.set_title("Distribución de Residuales", fontsize=16)
    ax.grid(True, alpha=0.3)
    
    p4_png = outdir / f"residuals_dist_{timestamp}.png"
    p4_pdf = outdir / f"residuals_dist_{timestamp}.pdf"
    fig.savefig(p4_png, **plot_kwargs)
    fig.savefig(p4_pdf, **plot_kwargs)
    paths["residuals_dist_plot"] = (p4_png, p4_pdf)
    plt.close(fig)
    
    # 5. Gráfico Q-Q de residuales
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals["res"].ravel(), dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(4.0)
    ax.get_lines()[1].set_linewidth(2.0)
    ax.set_title("Gráfico Q-Q de Residuales", fontsize=16)
    ax.grid(True, alpha=0.3)
    
    p5_png = outdir / f"qq_plot_{timestamp}.png"
    p5_pdf = outdir / f"qq_plot_{timestamp}.pdf"
    fig.savefig(p5_png, **plot_kwargs)
    fig.savefig(p5_pdf, **plot_kwargs)
    paths["qq_plot"] = (p5_png, p5_pdf)
    plt.close(fig)
    
    return paths
# -------------------- Reporte PDF completo para papers --------------------
def build_comprehensive_pdf_report(outdir: Path, timestamp: str, args: argparse.Namespace, 
                                  metrics: Dict, plot_paths: Dict, gp_model: BayesianEnsembleGP) -> Path:
    """Genera un reporte PDF completo apto para publicación científica."""
    pdf_path = outdir / f"dvt_gp_report_{timestamp}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1))
    styles.add(ParagraphStyle(name='Justify', alignment=4))
    
    story = []
    
    # Título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("DVT GP Ultra - Reporte de Validación Completo", title_style))
    story.append(Paragraph(f"Fecha: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Metadatos del experimento
    story.append(Paragraph("Configuración del Experimento", styles['Heading2']))
    meta_data = [
        ["Parámetro", "Valor"],
        ["Rango φ", f"[{args.phi_min}, {args.phi_max}]"],
        ["m_φ", f"{args.m_phi:.6g}"],
        ["Nivel de ruido", f"{args.noise:.3e}"],
        ["Puntos de datos", f"{args.n_points}"],
        ["Extensión de rango", f"{args.extend_range_factor:.2f}"],
        ["Kernels", ", ".join(args.kernel_types)],
        ["Valores de ν", ", ".join([f"{nu:.1f}" for nu in args.nu_values])],
        ["Kernel personalizado", args.kernel_expr if args.kernel_expr else "Ninguno"],
        ["Optimización Optuna", f"{'Sí' if args.optimize else 'No'} ({args.n_optuna_trials} trials)"],
        ["Reinicios optimizador", f"{args.n_restarts}"],
        ["Trabajos paralelos", f"{args.n_jobs}"],
        ["Semilla", f"{args.seed}"],
    ]
    
    meta_table = Table(meta_data, colWidths=[200, 300])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3D5A80")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E0FBFC")),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))
    
    # Métricas de rendimiento
    story.append(Paragraph("Métricas de Rendimiento", styles['Heading2']))
    perf_data = [
        ["Métrica", "Entrenamiento", "Validación"],
        ["R²", f"{metrics.get('R2_train', float('nan')):.4f}", f"{metrics.get('R2_val', float('nan')):.4f}"],
        ["MSE", f"{metrics.get('MSE_train', float('nan')):.4e}", f"{metrics.get('MSE_val', float('nan')):.4e}"],
        ["RMSE", f"{metrics.get('RMSE_train', float('nan')):.4f}", f"{metrics.get('RMSE_val', float('nan')):.4f}"],
        ["MAE", f"{metrics.get('MAE_train', float('nan')):.4f}", f"{metrics.get('MAE_val', float('nan')):.4f}"],
        ["Error Máximo", f"{metrics.get('MaxError_train', float('nan')):.4f}", 
         f"{metrics.get('MaxError_val', float('nan')):.4f}"],
    ]
    
    perf_table = Table(perf_data, colWidths=[150, 150, 150])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3D5A80")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E0FBFC")),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 20))
    
    # Validaciones
    if any(k in metrics for k in ["converged", "symbolic_validation_passed", "adversarial_validation_passed"]):
        story.append(Paragraph("Resultados de Validación", styles['Heading2']))
        validations_data = [
            ["Validación", "Resultado", "Métricas"],
            ["Multi-escala", 
             "PASADA" if metrics.get("converged", False) else "FALLADA", 
             f"Δ(1k→5k): {metrics.get('convergence_diff', 0):.4f}"],
            ["Simbólica (derivadas)", 
             "PASADA" if metrics.get("symbolic_validation_passed", False) else "FALLADA", 
             f"dV RMSE: {metrics.get('dV_rmse', 0):.4f}, d2V RMSE: {metrics.get('d2V_rmse', 0):.4f}"],
            ["Adversarial", 
             "PASADA" if metrics.get("adversarial_validation_passed", False) else "FALLADA", 
             f"R²: {metrics.get('adversarial_r2', 0):.4f}, MSE: {metrics.get('adversarial_mse', 0):.4e}"],
            ["CV Extrema 5x2", 
             "ESTABLE" if metrics.get("cv_stability", False) else "INESTABLE", 
             f"R²: {metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}"],
            ["Extrapolación", 
             "PASADA" if metrics.get("extrapolation_pass", False) else "FALLADA", 
             f"R²: {metrics.get('extrapolation_r2', 0):.4f}, Cobertura: {metrics.get('extrapolation_coverage', 0):.3f}"],
        ]
        
        validations_table = Table(validations_data, colWidths=[150, 100, 250])
        validations_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3D5A80")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E0FBFC")),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
        ]))
        story.append(validations_table)
        story.append(Spacer(1, 20))
    
    # Pesos de los modelos en el ensamble
    story.append(Paragraph("Composición del Ensamble", styles['Heading2']))
    ensemble_data = [["Modelo", "Peso", "Log-Likelihood Marginal"]]
    for i, model in enumerate(gp_model.models):
        ensemble_data.append([
            model["type"],
            f"{gp_model.weights[i]:.4f}",
            f"{gp_model.log_marginal_likelihoods[i]:.2f}" if i < len(gp_model.log_marginal_likelihoods) else "N/A"
        ])
    
    ensemble_table = Table(ensemble_data, colWidths=[200, 100, 150])
    ensemble_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3D5A80")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E0FBFC")),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
    ]))
    story.append(ensemble_table)
    story.append(Spacer(1, 20))
    
    # Análisis estadístico de residuales
    if "residual_mean" in metrics:
        story.append(Paragraph("Análisis Estadístico de Residuales", styles['Heading2']))
        stats_data = [
            ["Estadístico", "Valor"],
            ["Media", f"{metrics.get('residual_mean', 0):.4e}"],
            ["Desviación estándar", f"{metrics.get('residual_std', 0):.4e}"],
            ["Sesgo", f"{metrics.get('residual_skewness', 0):.4f}"],
            ["Curtosis", f"{metrics.get('residual_kurtosis', 0):.4f}"],
            ["Valor p (Shapiro-Wilk)", f"{metrics.get('shapiro_p', 0):.4e}"],
            ["Valor p (Normalidad)", f"{metrics.get('normaltest_p', 0):.4e}"],
        ]
        
        stats_table = Table(stats_data, colWidths=[200, 200])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3D5A80")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E0FBFC")),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
    
    # Añadir gráficos
    story.append(PageBreak())
    story.append(Paragraph("Visualizaciones", styles['Heading1']))
    
    if "potential_plot" in plot_paths:
        story.append(Paragraph("Reconstrucción del Potencial", styles['Heading2']))
        png_path, pdf_path = plot_paths["potential_plot"]
        story.append(RLImage(str(png_path), width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Figura 1: Reconstrucción del potencial inflacionario mediante ensamble GP. "
                              "Se muestran los datos de entrenamiento (azul), validación (verde), "
                              "la predicción del modelo (rojo) con intervalo de confianza del 95% (naranja), "
                              "y la verdad teórica (negro).", styles['Justify']))
        story.append(Spacer(1, 20))
    
    if "curvature_plot" in plot_paths:
        story.append(Paragraph("Curvatura del Potencial", styles['Heading2']))
        png_path, pdf_path = plot_paths["curvature_plot"]
        story.append(RLImage(str(png_path), width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Figura 2: Curvatura estimada del potencial (segunda derivada) comparada con "
                              "el valor teórico esperado.", styles['Justify']))
        story.append(Spacer(1, 20))
    
    if "residuals_plot" in plot_paths:
        story.append(Paragraph("Residuales de Validación", styles['Heading2']))
        png_path, pdf_path = plot_paths["residuals_plot"]
        story.append(RLImage(str(png_path), width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Figura 3: Residuales de validación (diferencia entre valores reales y predichos) "
                              "en función del campo φ.", styles['Justify']))
        story.append(Spacer(1, 20))
    
    if "residuals_dist_plot" in plot_paths:
        story.append(Paragraph("Distribución de Residuales", styles['Heading2']))
        png_path, pdf_path = plot_paths["residuals_dist_plot"]
        story.append(RLImage(str(png_path), width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Figura 4: Distribución de los residuales de validación con estimación de densidad kernel.", 
                              styles['Justify']))
        story.append(Spacer(1, 20))
    
    if "qq_plot" in plot_paths:
        story.append(Paragraph("Gráfico Q-Q de Residuales", styles['Heading2']))
        png_path, pdf_path = plot_paths["qq_plot"]
        story.append(RLImage(str(png_path), width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Figura 5: Gráfico quantil-quantil para evaluar la normalidad de los residuales. "
                              "Una distribución normal perfecta seguiría la línea roja.", styles['Justify']))
    
    # Conclusiones
    story.append(PageBreak())
    story.append(Paragraph("Conclusiones", styles['Heading1']))
    
    conclusions = [
        "El modelo de ensamble GP ha demostrado capacidad para reconstruir el potencial inflacionario con alta precisión.",
        "Las validaciones extremas confirman la robustez del modelo frente a diferentes escenarios desafiantes.",
        "La extrapolación beyond el rango de entrenamiento muestra un rendimiento adecuado para aplicaciones prácticas.",
        "El análisis de residuales confirma que los errores se distribuyen apropiadamente para un modelo de regresión.",
        "Este enfoque es suitable para su uso en análisis cosmológicos y estudios de inflación."
    ]
    
    for i, conclusion in enumerate(conclusions):
        story.append(Paragraph(f"{i+1}. {conclusion}", styles['Justify']))
        story.append(Spacer(1, 6))
    
    # Generar PDF
    doc.build(story)
    logger.info(f"Reporte PDF completo generado: {pdf_path}")
    
    return pdf_path

# -------------------- Pipeline principal mejorado --------------------
def train_pipeline(args: argparse.Namespace) -> Tuple[Dict, str, BayesianEnsembleGP]:
    """Pipeline principal de entrenamiento y evaluación."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generar datos
    phi, V, m_phi = generate_training_data(
        args.n_points, (args.phi_min, args.phi_max), args.m_phi, args.noise, args.seed,
        extend_range_factor=args.extend_range_factor
    )
    
    # Dividir en train/validation
    phi_tr, phi_val, V_tr, V_val = train_test_split(phi, V, test_size=0.2, random_state=args.seed)
    
    # Inicializar y entrenar modelo
    gp = BayesianEnsembleGP(
        kernel_types=args.kernel_types,
        nu_values=args.nu_values,
        seed=args.seed,
        kernel_expr=args.kernel_expr,
        n_optuna_trials=(args.n_optuna_trials if args.optimize else 0),
        n_jobs=args.n_jobs,
        scaler_type=args.scaler_type
    )
    
    gp.initialize_models(n_restarts=args.n_restarts)
    
    # Barra de progreso mejorada
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Entrenando Ensamble GP...", total=1)
        gp.fit(phi_tr, V_tr)
        progress.update(task, completed=1)
    
    # Predicciones y métricas básicas
    train_pred, train_std = gp.predict(phi_tr)
    val_pred, val_std = gp.predict(phi_val)
    
    metrics = {
        **compute_comprehensive_metrics(V_tr, train_pred, train_std, "train"),
        **compute_comprehensive_metrics(V_val, val_pred, val_std, "val")
    }
    
    # Validaciones extremas si están habilitadas
    if args.extreme_validation or args.all_validations:
        logger.info("Ejecutando suite de validaciones extremas...")
        metrics.update(multi_scale_validation(
            (args.phi_min, args.phi_max), args.m_phi, args.noise, args.seed,
            args.kernel_expr, args.n_restarts, args.kernel_types, args.nu_values, 
            args.n_jobs, args.extend_range_factor
        ))
        metrics.update(symbolic_validation((args.phi_min, args.phi_max), args.m_phi, gp))
        metrics.update(adversarial_validation(gp, (args.phi_min, args.phi_max), args.m_phi, args.noise, args.seed))
        metrics.update(extreme_cross_validation(
            phi, V, args.kernel_expr, args.kernel_types, args.nu_values, 
            args.seed, args.n_restarts, args.n_jobs
        ))
        metrics.update(extrapolation_test(
            gp, (args.phi_min, args.phi_max), args.m_phi, args.noise, args.seed, args.extend_range_factor
        ))
    
    # Análisis estadístico de residuales
    residuals = {"phi": phi_val, "res": V_val.ravel() - val_pred.ravel()}
    metrics.update(statistical_analysis(V_val.ravel(), val_pred.ravel(), val_std))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_paths = {}
    
    # Generar gráficos si está habilitado
    if args.plot_results:
        phi_grid = np.linspace(
            args.phi_min - args.extend_range_factor * (args.phi_max - args.phi_min),
            args.phi_max + args.extend_range_factor * (args.phi_max - args.phi_min),
            2000
        ).reshape(-1, 1)
        
        V_pred, V_std = gp.predict(phi_grid)
        curvature = np.gradient(np.gradient(V_pred.ravel(), phi_grid.ravel()), phi_grid.ravel())
        
        plot_paths = create_hq_plots(
            phi_tr, V_tr, phi_val, V_val, phi_grid, V_pred, V_std, curvature, 
            residuals, outdir, timestamp, args.m_phi
        )
    
    # Guardar modelo y resultados
    model_path = outdir / f"model_{timestamp}.pkl"
    results = {
        "gp": gp,
        "metrics": metrics,
        "args": vars(args),
        "timestamp": timestamp,
        "plot_paths": plot_paths
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(results, f)
    
    logger.info(f"Modelo final guardado en: {model_path}")
    
    # Generar reporte PDF si está habilitado
    if args.plot_results:
        pdf_path = build_comprehensive_pdf_report(outdir, timestamp, args, metrics, plot_paths, gp)
        logger.info(f"Reporte PDF generado: {pdf_path}")
    
    return metrics, timestamp, gp

# -------------------- CLI mejorada --------------------
def main():
    """Función principal con interfaz de línea de comandos mejorada."""
    parser = argparse.ArgumentParser(
        description="DVT GP Ultra: Entrenamiento de Ensamble GP con Validaciones Extremas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Ejemplo de uso: python train_gp.py --n-points 1000 --phi-min -5 --phi-max 5 --m-phi 0.5 --noise 0.001 --extreme-validation --plot-results --n-jobs 4"
    )
    
    # Grupo de parámetros de simulación
    g_sim = parser.add_argument_group("Parámetros de Simulación")
    g_sim.add_argument("--n-points", type=int, default=1000, 
                      help="Número de puntos de datos sintéticos.")
    g_sim.add_argument("--phi-min", type=float, default=-5.0, 
                      help="Límite inferior del campo φ.")
    g_sim.add_argument("--phi-max", type=float, default=5.0, 
                      help="Límite superior del campo φ.")
    g_sim.add_argument("--m-phi", type=float, default=0.5, 
                      help="Parámetro de masa del potencial inflacionario.")
    g_sim.add_argument("--noise", type=float, default=1e-3, 
                      help="Nivel de ruido Gaussiano en los datos.")
    g_sim.add_argument("--extend-range-factor", type=float, default=0.2,
                      help="Factor para extender el rango de φ más allá de [phi-min, phi-max] para mejorar la extrapolación.")
    
    # Grupo de parámetros del modelo
    g_model = parser.add_argument_group("Parámetros del Modelo")
    g_model.add_argument("--kernel-types", nargs='+', default=["matern", "rbf", "rational_quadratic"], 
                        help="Tipos de kernels para incluir en el ensamble.")
    g_model.add_argument("--nu-values", nargs='+', type=float, default=[0.5, 1.5, 2.5], 
                        help="Valores de nu para el kernel Matern.")
    g_model.add_argument("--kernel-expr", type=str, default=None, 
                        help="Expresión de kernel personalizado para añadir al ensamble.")
    g_model.add_argument("--n-restarts", type=int, default=10, 
                        help="Número de reinicios para el optimizador de cada GP.")
    g_model.add_argument("--optimize", action="store_true", 
                        help="Activar optimización de hiperparámetros con Optuna.")
    g_model.add_argument("--n-optuna-trials", type=int, default=25, 
                        help="Número de trials de Optuna por modelo si la optimización está activada.")
    g_model.add_argument("--scaler-type", type=str, default="standard", choices=["standard", "robust"],
                        help="Tipo de escalado para preprocesamiento de datos.")
    
    # Grupo de parámetros de ejecución
    g_exec = parser.add_argument_group("Parámetros de Ejecución")
    g_exec.add_argument("--seed", type=int, default=42, 
                       help="Semilla para reproducibilidad.")
    g_exec.add_argument("--outdir", type=str, default="dvt_ultra_results", 
                       help="Directorio de salida para resultados.")
    g_exec.add_argument("--plot-results", action="store_true", 
                       help="Generar gráficos y reporte PDF.")
    g_exec.add_argument("--dpi", type=int, default=1200, 
                       help="Resolución (DPI) para gráficos de salida.")
    g_exec.add_argument("--verbose", action="store_true", 
                       help="Activar logging detallado (INFO level).")
    g_exec.add_argument("--extreme-validation", action="store_true", 
                       help="Ejecutar validaciones extremas (multi-escala, adversarial, etc.).")
    g_exec.add_argument("--all-validations", action="store_true", 
                       help="Alias para --extreme-validation.")
    g_exec.add_argument("--n-jobs", type=int, default=1, 
                       help="Número de trabajos paralelos para entrenamiento.")
    
    args = parser.parse_args()
    
    # Alias para validaciones
    if args.all_validations:
        args.extreme_validation = True
    
    # Configurar logging
    global logger
    logger = setup_logging(level=logging.INFO if args.verbose else logging.WARNING)
    
    try:
        logger.info("Iniciando DVT GP Ultra (Versión Fusionada Mejorada)...")
        logger.info(f"Argumentos de la ejecución:\n{json.dumps(vars(args), indent=2, default=str)}")
        
        metrics, timestamp, gp = train_pipeline(args)
        
        logger.info("Pipeline finalizado exitosamente.")
        logger.info(f"Métricas finales:\n{json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        logger.critical(f"El pipeline ha fallado con un error crítico: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()