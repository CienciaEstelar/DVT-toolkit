# dvt/symbolic.py (Versión Adaptada y Robusta)
"""
Construcción simbólica del núcleo DVT (ecuaciones de campo) y
exportación de las funciones numéricas *phi_ddot_func* y *a_ddot_func*.

Esta versión ha sido adaptada para garantizar la máxima estabilidad y correctitud
en conjunto con los rangos de prior ampliados del nuevo mcmc.py.

Características clave
--------------------
* Derivación automática y garantizada mediante sp.euler_equations.
* Simplificación numéricamente estable con sp.cancel para expresiones racionales.
* Lambdify robusto que convierte expresiones a escalares float finitos.
* Caché cloudpickle para arranques subsecuentes ultrarrápidos.
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import sympy as sp

from .config import logger
try:
    import cloudpickle as pickle
except ImportError:
    import pickle

###############################################################################
# 1 ▪ Símbolos globales y archivo de caché
###############################################################################

# Variables simbólicas usadas en todas las expresiones.
_DEF_VARS = [sp.Symbol(s) for s in ("phi", "phi_dot", "a", "a_dot", "G", "xi")]

CACHE_SYM: Path = Path(__file__).with_suffix(".symcache")

###############################################################################
# 2 ▪ Conversor robusto -> float
###############################################################################

def _scalarizer(f: Callable[..., float]) -> Callable[..., float]:
    """Envuelve `f` para que siempre devuelva un *float* finito o `np.nan`."""

    def _wrap(*args):
        out = f(*args)
        if isinstance(out, np.ndarray) and out.shape == ():
            out = out.item()
        if isinstance(out, sp.Expr):
            if out is sp.zoo or out is sp.oo or out is -sp.oo:
                return np.inf * np.sign(float(out.evalf(1)))
            try:
                return float(out.evalf())
            except (TypeError, ValueError):
                return np.nan
        if isinstance(out, np.ndarray) and out.dtype == object:
            vfunc = np.vectorize(
                lambda z: float(z.evalf()) if isinstance(z, sp.Expr) else z,
                otypes=[float],
            )
            arr = out.ravel()
            return float(vfunc(arr)[0])
        try:
            return float(out)
        except Exception:
            return np.nan

    return _wrap

###############################################################################
# 3 ▪ Construcción de las ecuaciones (Versión Robusta)
###############################################################################

def _build_symbolic() -> Tuple[sp.Expr, sp.Expr]:
    """
    Deriva y devuelve las expresiones SymPy para ddot(phi) y ddot(a)
    usando el método de Euler-Lagrange para máxima robustez.
    """
    # --- Definición de Símbolos y Funciones ---
    t = sp.Symbol("t")
    phi_func = sp.Function("phi")(t)
    a_func = sp.Function("a")(t)
    V_func = sp.Function("V")(phi_func)
    G, xi = sp.symbols("G xi")
    
    # Derivadas
    phi_dot = sp.diff(phi_func, t)
    phi_ddot = sp.diff(phi_dot, t)
    a_dot = sp.diff(a_func, t)
    a_ddot = sp.diff(a_dot, t)

    # --- Construcción del Lagrangiano ---
    # Escalar de Ricci para un universo FLRW plano (k=0)
    R = 6 * (a_ddot / a_func + (a_dot / a_func)**2)

    # Lagrangiano con acoplamiento no mínimo
    L = R / (16 * sp.pi * G) - (0.5 * phi_dot**2 - V_func) - 0.5 * xi * R * phi_func**2

    # --- Derivación Automática con Euler-Lagrange ---
    # Este método es superior a la derivación manual, ya que es menos propenso a errores.
    eqs = sp.euler_equations(L, [phi_func, a_func], t)

    # --- Resolución del Sistema de Ecuaciones ---
    # Resolvemos el sistema de EDOs para las segundas derivadas
    soluciones = sp.solve(eqs, [phi_ddot, a_ddot], dict=True)
    if not soluciones:
        raise RuntimeError("No se pudo resolver el sistema de EDOs para phi_ddot y a_ddot.")
    
    sol = soluciones[0]
    e_phi = sol[phi_ddot]
    e_a = sol[a_ddot]

    # --- Simplificación Numéricamente Estable ---
    # Usamos sp.cancel para obtener una forma p/q, que es más estable
    # para funciones racionales. Esto es clave para la estabilidad numérica.
    e_phi_cancelled = sp.cancel(e_phi)
    e_a_cancelled = sp.cancel(e_a)
    
    # NOTA SOBRE LA ESTABILIDAD:
    # Las expresiones resultantes tendrán un denominador común proporcional a:
    # (1 - 8*pi*G*xi*phi^2).
    # Este término puede anularse, creando una singularidad física.
    # Este módulo, symbolic.py, entrega la forma matemática correcta.
    # La responsabilidad de manejar esta singularidad recae en el módulo numérico
    # (solver.py), que ya implementa chequeos para detener la integración
    # si el sistema se acerca a este punto crítico.
    
    # Reemplazamos las funciones por símbolos para el lambdify
    phi_sym, phi_dot_sym, a_sym, a_dot_sym = _DEF_VARS[:4]
    V_sym, dV_sym = sp.symbols("V dV")
    
    subs_dict = {
        phi_func: phi_sym, a_func: a_sym,
        phi_dot: phi_dot_sym, a_dot: a_dot_sym,
        sp.diff(V_func, phi_func): dV_sym, V_func: V_sym,
    }

    return e_phi_cancelled.subs(subs_dict), e_a_cancelled.subs(subs_dict)

###############################################################################
# 4 ▪ Creación de funciones numéricas + caché
###############################################################################

def _make_lambdify(e_phi: sp.Expr, e_a: sp.Expr) -> Tuple[Callable[..., float], Callable[..., float]]:
    """Lambdify robusto y envuelto con _scalarizer."""
    if e_phi.has(sp.Derivative) or e_a.has(sp.Derivative):
        raise RuntimeError("Las expresiones todavía contienen derivadas simbólicas no resueltas.")

    variables = _DEF_VARS + [sp.Symbol("V"), sp.Symbol("dV")]
    try:
        raw_phi = sp.lambdify(variables, e_phi, modules=["numpy", "sympy", "mpmath"])
        raw_a = sp.lambdify(variables, e_a, modules=["numpy", "sympy", "mpmath"])
    except Exception as exc:
        logger.error("Fallo al lambdificar: %s", exc)
        raise

    return _scalarizer(raw_phi), _scalarizer(raw_a)


def _load_or_build_funcs() -> Tuple[Callable[..., float], Callable[..., float]]:
    """Carga las funciones de `.symcache` o las reconstruye."""
    if CACHE_SYM.exists() and CACHE_SYM.stat().st_size > 0:
        try:
            with CACHE_SYM.open("rb") as fh:
                logger.info("⏩ Cargando ecuaciones cacheadas de %s", CACHE_SYM.name)
                return pickle.load(fh)
        except Exception as exc:
            warnings.warn(f"Fallo al leer la caché {CACHE_SYM}: {exc} — regenerando", RuntimeWarning)

    logger.info("🛠 Generando ecuaciones simbólicas DVT (primera ejecución)...")
    e_phi, e_a = _build_symbolic()
    funcs = _make_lambdify(e_phi, e_a)

    try:
        with CACHE_SYM.open("wb") as fh:
            pickle.dump(funcs, fh)
        logger.info("Ecuaciones cacheadas en %s para futuras ejecuciones.", CACHE_SYM)
    except Exception as exc:
        warnings.warn(f"No se pudo guardar la caché {CACHE_SYM}: {exc}")

    return funcs

###############################################################################
# 5 ▪ API pública
###############################################################################

phi_ddot_func, a_ddot_func = _load_or_build_funcs()

__all__ = [
    "phi_ddot_func",
    "a_ddot_func",
]
