# dvt/solver.py (VERSIÓN FINAL CON MANEJO DE ERRORES TOTALMENTE SILENCIOSO)
from __future__ import annotations
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Union
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from astropy import units as u
import logging
import time
from threading import Lock
from dataclasses import dataclass
from functools import lru_cache

# Configuración de logging
logger = logging.getLogger('DVT_SOLVER')
logger.setLevel(logging.INFO)

# =====================================================================
# ESTRUCTURAS DE DATOS
# =====================================================================
@dataclass
class CosmologicalParameters:
    H0: float         # Constante de Hubble (km/s/Mpc)
    ombh2: float      # Densidad bariónica Ω_b h²
    omch2: float      # Densidad materia oscura Ω_c h²
    phi0: float       # Valor inicial del campo escalar
    phidot0: float    # Derivada inicial del campo
    xi: float         # Acoplamiento no mínimo
    logA: float       # Log-amplitud del potencial (GP)
    logl: float       # Log-longitud de escala (GP)

class PotentialGP:
    def V(self, phi: float) -> float: ...

# Constantes físicas optimizadas
class PhysicalConstants:
    c = 299_792.458    # Velocidad de la luz (km/s)
    G = 6.67430e-11    # Constante gravitacional (m³/kg/s²)

# Límites numéricos
class SafetyLimits:
    MAX_PHI = 5.0
    MIN_SCALE_FACTOR = 1e-10
    SAFE_G_EPS = 1e-8
    DERIV_EPS = 1e-6
    OMEGA_TOL = 1e-3

# =====================================================================
# CACHÉ Y FUNCIONES CACHEADAS
# =====================================================================
_SOLUTION_CACHE: Dict[Tuple, Optional[Tuple]] = {}
_CACHE_LOCK = Lock()
_CACHE_MAX_SIZE = 100

@lru_cache(maxsize=100)
def phi_ddot_func(phi: float, phidot: float, a: float, adot: float,
                 G: float, xi: float, V: float, Vp: float) -> float:
    """Ecuación para φ̈ con cache LRU."""
    H = adot / max(a, 1e-16)
    return -3.0 * H * phidot - Vp - 6.0 * xi * (H**2) * phi

@lru_cache(maxsize=100)
def a_ddot_func(phi: float, phidot: float, a: float, adot: float,
               G: float, xi: float, V: float, Vp: float) -> float:
    """Ecuación para ä con cache LRU."""
    rho = 0.5 * phidot**2 + V
    p = 0.5 * phidot**2 - V
    return -(4.0 * np.pi * G / 3.0) * (rho + 3.0 * p) * a

# =====================================================================
# FUNCIÓN PRINCIPAL SOLVE_DVT
# =====================================================================
def solve_dvt(
    theta: Union[np.ndarray, CosmologicalParameters],
    gp: PotentialGP,
    z_max: float = 1100,
    fast_mode: bool = True
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float]]:
    """Integra las ecuaciones DVT con manejo de errores totalmente silenciado para MCMC."""
    try:
        params = (theta if isinstance(theta, CosmologicalParameters)
                 else CosmologicalParameters(*np.asarray(theta, dtype=float)))

        with _CACHE_LOCK:
            if (cached := _SOLUTION_CACHE.get(tuple(vars(params).values()))):
                return cached

        rtol = 1e-4 if fast_mode else 1e-6
        atol = 1e-6 if fast_mode else 1e-8
        n_points = 200 if fast_mode else 400

        a0 = 1.0 / (1.0 + z_max)
        h = params.H0 / 100.0
        Omega_m = (params.ombh2 + params.omch2) / h**2
        Omega_r = 4.15e-5 / h**2
        adot0 = params.H0 * a0 * np.sqrt(Omega_m * a0**-3 + Omega_r * a0**-4)
        y0 = np.array([params.phi0, params.phidot0, a0, adot0], dtype=float)

        def ode_system(t: float, y: np.ndarray):
            phi, phidot, a, adot = y
            if not (-SafetyLimits.MAX_PHI <= phi <= SafetyLimits.MAX_PHI) or a <= SafetyLimits.MIN_SCALE_FACTOR:
                return np.full(4, np.nan)

            try:
                V_val = gp.V(phi)
                eps = max(SafetyLimits.DERIV_EPS, SafetyLimits.DERIV_EPS * abs(phi))
                Vp = (gp.V(phi + eps) - gp.V(phi - eps)) / (2 * eps)

                return np.array([
                    phidot * (a / max(adot, 1e-16)),
                    phi_ddot_func(phi, phidot, a, adot, PhysicalConstants.G, params.xi, V_val, Vp),
                    a,
                    a_ddot_func(phi, phidot, a, adot, PhysicalConstants.G, params.xi, V_val, Vp)
                ])
            except (ValueError, TypeError):
                return np.full(4, np.nan)

        sol = solve_ivp(
            ode_system,
            (np.log(a0), 0.0),
            y0,
            method='BDF',
            t_eval=np.linspace(np.log(a0), 0.0, n_points),
            rtol=rtol,
            atol=atol,
            dense_output=False
        )

        if sol.success:
            z = (1/np.exp(sol.t) - 1)
            H = sol.y[3]/sol.y[2]

            valid = np.isfinite(z) & np.isfinite(H)
            z, H = z[valid], H[valid]

            if len(z) < 2: return None

            sort_idx = np.argsort(z)
            z, H = z[sort_idx], H[sort_idx]

            unique_mask = np.concatenate(([True], np.diff(z) > 1e-10))
            z, H = z[unique_mask], H[unique_mask]

            if len(z) < 2 or np.any(H <= 0):
                return None

            result = (z, H, params.H0, params.ombh2, params.omch2)
            with _CACHE_LOCK:
                if len(_SOLUTION_CACHE) >= _CACHE_MAX_SIZE:
                    _SOLUTION_CACHE.clear()
                _SOLUTION_CACHE[tuple(vars(params).values())] = result
            return result

    # --- INICIO DE LA MODIFICACIÓN: SILENCIO TOTAL DE ERRORES ---
    except Exception:
        # Atrapa cualquier excepción que ocurra durante la integración
        # y simplemente retorna None sin registrar ningún warning.
        # Esto es crucial para mantener una salida limpia en el MCMC.
        pass
    # --- FIN DE LA MODIFICACIÓN ---
    
    return None

__all__ = ['solve_dvt', 'CosmologicalParameters']