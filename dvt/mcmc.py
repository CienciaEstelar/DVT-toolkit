#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# dvt/mcmc.py — Versión Final Integrada y Robusta para Pipeline DVT
# ==============================================================================
# Driver MCMC (emcee v3) optimizado para el pipeline completo de Dinámica del Vacío Tensorial.
# - Integración perfecta con PotentialGP, Likelihood y el resto del pipeline
# - Inicialización robusta de walkers dentro de priors físicos bien definidos
# - Multiprocesamiento seguro con control inteligente de workers
# - Reanudación de cadenas compatible con emcee v3+
# - Diagnóstico completo de convergencia y tasas de aceptación
# - Manejo robusto de errores y logging detallado
# - Modo smoke test para verificación rápida del pipeline
# ==============================================================================

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from multiprocessing import get_context, cpu_count
from typing import Tuple, Optional, Dict, Any

import numpy as np
import emcee

# Importaciones del paquete DVT con fallback para smoke test
try:
    from .potential import PotentialGP
    from .likelihood import Likelihood
    from .config import logger, SEED
    from .cosmology import CosmoHelper
    from .data import load_data
except ImportError as e:
    # ==========================================================================
    # BLOQUE DE DIAGNÓSTICO INTEGRADO
    # ==========================================================================
    # Fallback robusto para ejecución independiente y smoke tests
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("DVT_MCMC_FALLBACK")
    logger.error("="*80)
    logger.error(" ATENCIÓN: MODO FALLBACK (DE PRUEBA) ACTIVADO EN MCMC.PY")
    logger.error(f"  Error de importación original: {e}")
    logger.error("  Se usarán clases simuladas. ¡El cálculo será RÁPIDO pero INCORRECTO!")
    logger.error("  Esto ocurre si ejecutas el script de una manera que Python no encuentra")
    logger.error("  los otros módulos del paquete (potential.py, likelihood.py, etc.)")
    logger.error("="*80)
    # ==========================================================================
    
    # Constantes por defecto para el pipeline
    SEED = 42
    
    # Clases mínimas para smoke test
    class PotentialGP:
        def __init__(self): 
            self.hyper = np.array([0.0, 0.0])
            self.phi_nodes = np.linspace(-5, 5, 50)
            self.V_nodes = 0.5 * self.phi_nodes**2
            
        def update(self, new_theta): 
            self.hyper = np.asarray(new_theta).copy()
            
        def V(self, phi: np.ndarray) -> np.ndarray: 
            return 0.5 * (np.asarray(phi).ravel() ** 2)
            
        def physical_potential(self, phi: np.ndarray) -> np.ndarray:
            return self.V(phi) * (3.0 * (67.4**2) / (8 * np.pi * 6.67430e-11)) / (3.086e22**2)
    
    class Likelihood:
        def __init__(self, data): 
            self.data = data
            self.cosmo_helper = CosmoHelper()
            
        def total(self, theta, gp: PotentialGP) -> float:
            # Simulación simplificada para smoke test
            H0, ombh2, omch2, phi0, phidot0, xi, logA, logL = theta[:8]
            
            # Validación física básica
            if not (50.0 < H0 < 90.0) or not (0.018 < ombh2 < 0.026) or not (0.10 < omch2 < 0.14):
                return -np.inf
                
            # Cálculo simplificado de chi2 para pruebas
            chi2 = 0.0
            
            # Componente CMB (aproximación)
            if 'cmb' in self.data:
                chi2 += ((H0 - 67.4) / 0.5)**2 + ((ombh2 - 0.0224) / 0.0001)**2 + ((omch2 - 0.120) / 0.001)**2
            
            # Componente SN (aproximación)
            if 'sn' in self.data:
                chi2 += ((H0 - 73.0) / 1.0)**2
                
            return -0.5 * chi2
    
    class CosmoHelper:
        def comoving_distance(self, z, H0, ombh2, omch2, **kwargs):
            # Aproximación plana para smoke test
            Om_m = (ombh2 + omch2) / (H0/100)**2
            return (3000 / H0) * np.log(1 + z) if Om_m < 0.1 else (2000 / H0) * z

# Exponer API del módulo
__all__ = ["DVT_MCMC", "run_mcmc_sampling"]

# ---------------------------------------------------------------------------
# Clase principal: DVT_MCMC
# ---------------------------------------------------------------------------
class DVT_MCMC:
    """
    Clase que orquesta el muestreo MCMC para el pipeline DVT completo.
    Integra Perfectamente con PotentialGP, Likelihood y demás componentes.
    """
    
    # Parámetros cosmológicos estándar como referencia
    COSMOLOGICAL_PARAMS = {
        'H0': (67.4, 50.0, 90.0),
        'ombh2': (0.0224, 0.018, 0.026),
        'omch2': (0.120, 0.10, 0.14),
        'phi0': (0.0, -5.0, 5.0),
        'phidot0': (0.0, -5.0, 5.0),
        'xi': (0.1, 0.0, 1.0),
        'logA': (0.0, -10.0, 10.0),
        'logL': (0.0, -10.0, 10.0)
    }
    
    def __init__(
        self,
        data: Dict[str, Any],
        *,
        walkers: int = 50,
        steps: int = 2000,
        use_pool: bool = False,
        convergence_threshold: float = 50.0,
        target_acceptance: Tuple[float, float] = (0.2, 0.5),
        ndim: int = 8,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Inicializa el sampler MCMC para el pipeline DVT.
        
        Args:
            data: Diccionario con datos observacionales
            walkers: Número de walkers
            steps: Pasos por cadena
            use_pool: Usar multiprocesamiento
            convergence_threshold: Umbral de convergencia (múltiplo de tau)
            target_acceptance: Rango óptimo de aceptación
            ndim: Dimensionalidad del espacio de parámetros
            seed: Semilla para reproducibilidad
            output_dir: Directorio de salida para resultados
        """
        self.data = data
        self.gp = PotentialGP()
        self.like = Likelihood(data)
        self.walkers = int(walkers)
        self.steps = int(steps)
        self.use_pool = bool(use_pool)
        self.convergence_threshold = float(convergence_threshold)
        self.target_acceptance = tuple(target_acceptance)
        self.ndim = int(ndim)
        self.seed = seed if seed is not None else SEED
        self.output_dir = output_dir or Path("mcmc_results")
        self.sampler: Optional[emcee.EnsembleSampler] = None
        self._last_theta_gp = None
        
        # Configurar directorio de salida
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"DVT_MCMC inicializado: {self.walkers} walkers, {self.steps} steps, "
                   f"{self.ndim} params, seed={self.seed}")
        logger.info(f"Output directory: {self.output_dir.resolve()}")

    @staticmethod
    def _log_prior(theta: np.ndarray) -> float:
        """
        Prior plano con límites físicos bien definidos para cada parámetro.
        Coherente con los requerimientos del pipeline completo.
        """
        theta = np.asarray(theta).ravel()
        if theta.size < 8:
            return -np.inf
            
        H0, ombh2, omch2, phi0, phidot0, xi, logA, logL = theta[:8]
        
        # Límites físicos estrictos basados en cosmología observacional
        in_bounds = (
            (50.0 < H0 < 90.0) and                    # H0 razonable
            (0.018 < ombh2 < 0.026) and               # Densidad bariónica
            (0.10 < omch2 < 0.14) and                 # Densidad materia oscura
            (-5.0 < phi0 < 5.0) and                   # Campo escalar inicial
            (-5.0 < phidot0 < 5.0) and                # Derivada del campo
            (0.0 <= xi <= 1.0) and                    # Parámetro de acoplamiento
            (-10.0 < logA < 10.0) and                 # Amplitud del GP
            (-10.0 < logL < 10.0)                     # Longitud de escala del GP
        )
        return 0.0 if in_bounds else -np.inf

    def _log_prob(self, theta: np.ndarray) -> float:
        """
        Función de log-probabilidad: prior + likelihood.
        Optimizada para integración con el pipeline DVT completo.
        """
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        try:
            # Actualiza el GP solo si los hiperparámetros han cambiado
            gp_theta = theta[6:8]
            if (self._last_theta_gp is None or 
                not np.allclose(gp_theta, self._last_theta_gp, rtol=1e-10, atol=1e-10)):
                self.gp.update(gp_theta)
                self._last_theta_gp = gp_theta.copy()
            
            # Calcula el likelihood usando la clase Likelihood del pipeline
            log_likelihood = self.like.total(theta, self.gp)
            
            if not np.isfinite(log_likelihood):
                return -np.inf
            
            return lp + log_likelihood
            
        except Exception as e:
            logger.debug(f"Error en _log_prob: {e}")
            return -np.inf

    def _initialize_walkers(self) -> np.ndarray:
        """
        Inicializa walkers alrededor de valores cosmológicos razonables.
        Garantiza que todos comiencen en posiciones físicamente válidas.
        """
        # Valores centrales basados en cosmología estándar
        center = np.array([
            67.4,      # H0 (Planck)
            0.0224,    # ombh2
            0.120,     # omch2
            0.0,       # phi0
            0.0,       # phidot0
            0.1,       # xi
            0.0,       # logA
            0.0        # logL
        ])
        
        # Dispersión controlada para exploración eficiente
        scale = np.array([
            2.0,       # H0
            0.001,     # ombh2
            0.005,     # omch2
            0.5,       # phi0
            0.5,       # phidot0
            0.05,      # xi
            1.0,       # logA
            1.0        # logL
        ])
        
        rng = np.random.default_rng(self.seed)
        pos = center + scale * rng.standard_normal(size=(self.walkers, self.ndim))
        
        # Validación robusta de todas las posiciones iniciales
        valid_mask = np.array([np.isfinite(self._log_prior(p)) for p in pos])
        invalid_count = np.sum(~valid_mask)
        
        if invalid_count > 0:
            # Regenerar walkers inválidos
            for i in range(self.walkers):
                if not valid_mask[i]:
                    attempts = 0
                    while attempts < 100:
                        new_pos = center + scale * rng.standard_normal(size=self.ndim)
                        if np.isfinite(self._log_prior(new_pos)):
                            pos[i] = new_pos
                            valid_mask[i] = True
                            break
                        attempts += 1
            
            # Verificación final
            valid_mask = np.array([np.isfinite(self._log_prior(p)) for p in pos])
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                raise RuntimeError(f"{invalid_count} walkers no pudieron ser inicializados dentro del prior.")
        
        logger.info(f"Walkers inicializados exitosamente. Todos dentro de prior físico.")
        return pos

    def _check_acceptance(self) -> None:
        """Diagnóstico detallado de tasas de aceptación."""
        if self.sampler is None:
            return
            
        acceptance_fraction = self.sampler.acceptance_fraction
        mean_acceptance = np.mean(acceptance_fraction)
        std_acceptance = np.std(acceptance_fraction)
        
        logger.info(f"Tasa de aceptación: {mean_acceptance:.3f} ± {std_acceptance:.3f}")
        
        # Verificación de rango óptimo
        low, high = self.target_acceptance
        if not (low <= mean_acceptance <= high):
            logger.warning(f"Tasa de aceptación fuera del rango óptimo [{low}, {high}]")
        
        # Detección de walkers problemáticos
        low_accept = np.sum(acceptance_fraction < 0.1)
        high_accept = np.sum(acceptance_fraction > 0.8)
        
        if low_accept > 0:
            logger.warning(f"{low_accept} walkers con aceptación < 10%")
        if high_accept > 0:
            logger.warning(f"{high_accept} walkers con aceptación > 80%")

    def _check_convergence(self) -> bool:
        """Verificación robusta de convergencia usando autocorrelación."""
        if self.sampler is None or self.sampler.iteration < 100:
            return False
        
        try:
            tau = self.sampler.get_autocorr_time(tol=0.01, quiet=True)
            max_tau = np.max(tau)
            converged = np.all(self.sampler.iteration > self.convergence_threshold * tau)
            
            if converged:
                logger.info(f"✅ Convergencia alcanzada (τ_max = {max_tau:.1f}, "
                           f"iterations = {self.sampler.iteration})")
                return True
            else:
                logger.info(f"⏳ No convergido (τ_max = {max_tau:.1f}, "
                           f"needed = {self.convergence_threshold * max_tau:.0f})")
                return False
                
        except (emcee.autocorr.AutocorrError, ValueError) as e:
            logger.warning(f"Autocorrelación no calculable: {e}")
            return False

    def _save_results(self) -> None:
        """Guarda resultados del MCMC de forma robusta."""
        if self.sampler is None:
            return
            
        try:
            # Cadena completa
            chain = self.sampler.get_chain()
            np.save(self.output_dir / "chain.npy", chain)
            
            # Log-probabilidades
            log_prob = self.sampler.get_log_prob()
            np.save(self.output_dir / "log_prob.npy", log_prob)
            
            # Parámetros de aceptación
            acceptance = self.sampler.acceptance_fraction
            np.save(self.output_dir / "acceptance.npy", acceptance)
            
            # Metadatos
            metadata = {
                'walkers': self.walkers,
                'steps': self.steps,
                'ndim': self.ndim,
                'seed': self.seed,
                'convergence_threshold': self.convergence_threshold,
                'target_acceptance': self.target_acceptance,
                'iteration': self.sampler.iteration,
                'mean_acceptance': float(np.mean(acceptance)),
            }
            
            with open(self.output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Resultados guardados en {self.output_dir.resolve()}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")

    def run(self) -> "DVT_MCMC":
        """
        Ejecuta el muestreo MCMC completo con manejo robusto de errores.
        Integrado perfectamente con el pipeline DVT.
        """
        initial_state = self._initialize_walkers()
        pool = None

        # Configurar multiprocesamiento
        if self.use_pool:
            n_procs = min(self.walkers, cpu_count() or 1)
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=n_procs)
            logger.info(f"Multiprocesamiento activado con {n_procs} workers")

        try:
            # Inicializar sampler
            self.sampler = emcee.EnsembleSampler(
                self.walkers, self.ndim, self._log_prob, pool=pool
            )
            
            # Primera pasada
            logger.info("Iniciando muestreo MCMC...")
            self.sampler.run_mcmc(initial_state, self.steps, progress=True, store=True)
            
            # Diagnóstico después de primera pasada
            self._check_acceptance()
            
            # Segunda pasada si no convergió
            if not self._check_convergence():
                logger.info("Iniciando segunda pasada para mejorar convergencia...")
                last_pos = self.sampler.get_last_sample().coords
                self.sampler.run_mcmc(last_pos, self.steps // 2, progress=True, store=True)
                
                # Diagnóstico final
                self._check_acceptance()
                self._check_convergence()
            
            # Guardar resultados
            self._save_results()
            logger.info("Muestreo MCMC completado exitosamente")
            
            return self

        except KeyboardInterrupt:
            logger.warning("MCMC interrumpido por el usuario")
            if self.sampler is not None:
                self._save_results()
            raise
            
        except Exception as e:
            logger.error(f"Error durante el muestreo MCMC: {e}")
            if self.sampler is not None:
                self._save_results()
            raise
            
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                logger.info("Pool de multiprocesamiento cerrado")

    def get_chain(self, **kwargs):
        """Obtiene la cadena MCMC."""
        if self.sampler is None:
            raise RuntimeError("El muestreo no ha sido ejecutado.")
        return self.sampler.get_chain(**kwargs)
        
    def get_autocorr_time(self, **kwargs):
        """Calcula el tiempo de autocorrelación."""
        if self.sampler is None:
            raise RuntimeError("El muestreo no ha sido ejecutado.")
        return self.sampler.get_autocorr_time(**kwargs)
        
    def get_results(self) -> Dict[str, Any]:
        """Retorna todos los resultados en un diccionario."""
        if self.sampler is None:
            raise RuntimeError("El muestreo no ha sido ejecutado.")
            
        return {
            'chain': self.get_chain(),
            'log_prob': self.sampler.get_log_prob(),
            'acceptance': self.sampler.acceptance_fraction,
            'autocorr_time': self.get_autocorr_time(),
            'last_sample': self.sampler.get_last_sample()
        }


# ---------------------------------------------------------------------------
# Función de conveniencia para el pipeline
# ---------------------------------------------------------------------------
def run_mcmc_sampling(
    data: Optional[Dict[str, Any]] = None,
    walkers: int = 50,
    steps: int = 2000,
    use_pool: bool = False,
    output_dir: Optional[Path] = None,
    **kwargs
) -> DVT_MCMC:
    """
    Función de conveniencia para ejecutar el muestreo MCMC desde el pipeline.
    
    Args:
        data: Datos observacionales (si None, se cargan automáticamente)
        walkers: Número de walkers
        steps: Pasos por cadena
        use_pool: Usar multiprocesamiento
        output_dir: Directorio de salida
        **kwargs: Argumentos adicionales para DVT_MCMC
        
    Returns:
        Instancia de DVT_MCMC con los resultados
    """
    if data is None:
        # Cargar datos por defecto del pipeline
        try:
            data = load_data(datasets=['cmb', 'sn', 'bao'])
            logger.info("Datos observacionales cargados automáticamente")
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos: {e}. Usando datos de prueba.")
            data = {'smoke_test': True}
    
    mcmc = DVT_MCMC(
        data=data,
        walkers=walkers,
        steps=steps,
        use_pool=use_pool,
        output_dir=output_dir,
        **kwargs
    )
    
    return mcmc.run()


# ---------------------------------------------------------------------------
# Modo smoke test para verificación del pipeline
# ---------------------------------------------------------------------------
def _generate_smoke_data(n=100, seed=42):
    """Genera datos sintéticos para pruebas del pipeline."""
    rng = np.random.default_rng(seed)
    
    # Datos CMB-like
    cmb_data = {
        'H0': 67.4 + rng.normal(0, 0.5),
        'ombh2': 0.0224 + rng.normal(0, 0.0001),
        'omch2': 0.120 + rng.normal(0, 0.001)
    }
    
    # Datos SN-like
    z_sn = np.linspace(0.01, 1.0, n)
    mu = 5 * np.log10(3000 * z_sn) + 25  # Aproximación simple
    mu_err = 0.1 + 0.05 * rng.random(n)
    mu_obs = mu + rng.normal(0, mu_err)
    
    sn_data = {
        'z': z_sn,
        'mu': mu_obs,
        'mu_err': mu_err
    }
    
    return {'cmb': cmb_data, 'sn': sn_data, 'smoke_test': True}


if __name__ == "__main__":
    # Smoke test completo del pipeline
    logger.info("Iniciando smoke test del pipeline DVT-MCMC...")
    
    smoke_data = _generate_smoke_data()
    mcmc = DVT_MCMC(smoke_data, walkers=12, steps=200, use_pool=False, ndim=8)
    
    try:
        results = mcmc.run()
        logger.info("✅ Smoke test completado exitosamente")
        
        # Análisis básico de resultados
        chain = mcmc.get_chain()
        logger.info(f"Dimensión de la cadena: {chain.shape}")
        logger.info(f"Tasa de aceptación media: {np.mean(mcmc.sampler.acceptance_fraction):.3f}")
        
    except Exception as e:
        logger.error(f"❌ Smoke test falló: {e}")
        sys.exit(1)
# ==============================================================================