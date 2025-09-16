# genesis_modular/__init__.py
"""API principal del paquete Genesis Modular DVT.

Este __init__ realiza *lazy imports* de los componentes principales para:
- Evitar imports pesados/circulares en el momento de importación del paquete.
- Hacer visible una API limpia (atributos accesibles como `genesis_modular.PotentialGP`).
- Ofrecer una comprobación de imports en modo desarrollo si se activa GENESIS_DEV=1.
"""

from __future__ import annotations

import importlib
import os
import logging
from types import ModuleType
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Versión del paquete
__version__ = "0.2.2"  # Actualizado para reflejar la corrección

# Lista pública de nombres exportados por el paquete
__all__ = [
    # Funciones principales
    "load_data", "solve_dvt",
    # Clases clave
    "PotentialGP", "Likelihood", "DVT_MCMC", "CosmoHelper", "BayesianEnsembleGP",
    # Utilidades
    "logger", "DATA_PATH", "USE_JAX",
    # Constantes
    "__version__", "PhysicalConstants", "SafetyLimits"
]

# Mapa de export_name -> (submodule_relative, attribute_name)
# submodule_relative se importa con importlib.import_module(submodule, package=__name__)
_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "load_data": (".dvt.data", "load_data"),
    "PotentialGP": (".dvt.potential", "PotentialGP"),
    "solve_dvt": (".dvt.solver", "solve_dvt"),
    "CosmoHelper": (".dvt.cosmology", "CosmoHelper"),
    "Likelihood": (".dvt.likelihood", "Likelihood"),
    "DVT_MCMC": (".dvt.mcmc", "DVT_MCMC"),
    "BayesianEnsembleGP": (".train_gp", "BayesianEnsembleGP"),  # <-- CORRECCIÓN AÑADIDA AQUÍ
    "logger": (".dvt.config", "logger"),
    "DATA_PATH": (".dvt.config", "DATA_PATH"),
    "USE_JAX": (".dvt.config", "USE_JAX"),
    "PhysicalConstants": (".dvt.solver", "PhysicalConstants"),
    "SafetyLimits": (".dvt.solver", "SafetyLimits"),
}

# Cache local de módulos ya importados (para evitar re-imports repetidos)
_IMPORTED_MODULES: Dict[str, ModuleType] = {}


def _lazy_import(name: str):
    """Importar dinámicamente name según _EXPORT_MAP y cachear en globals()."""
    if name not in _EXPORT_MAP:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

    submodule_rel, attr = _EXPORT_MAP[name]
    try:
        # Importar submódulo relativo (package=__name__)
        if submodule_rel in _IMPORTED_MODULES:
            mod = _IMPORTED_MODULES[submodule_rel]
        else:
            mod = importlib.import_module(submodule_rel, package=__name__)
            _IMPORTED_MODULES[submodule_rel] = mod
        obj = getattr(mod, attr)
    except Exception as e:
        # Registrar y volver a lanzar AttributeError para que el import falle de forma clara
        logger.debug(f"Error importando {name} desde {submodule_rel}: {e}")
        raise AttributeError(f"No se pudo importar '{name}' desde '{submodule_rel}': {e}") from e

    # Exponer en el namespace del paquete para accesos futuros directos
    globals()[name] = obj
    return obj


def __getattr__(name: str):
    """Permite acceso lazy: from genesis_modular import PotentialGP (lo importa al pedirlo)."""
    if name == "__version__":
        return __version__
    if name in _EXPORT_MAP:
        return _lazy_import(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Mejora autocompletado: lista nombres públicos + dinámicos."""
    return sorted(list(globals().keys()) + list(_EXPORT_MAP.keys()))


def _check_imports_dev():
    """Comprobación de imports (solo en modo desarrollo)."""
    errors = {}
    for name in sorted(_EXPORT_MAP.keys()):
        try:
            _lazy_import(name)
        except Exception as e:
            errors[name] = str(e)
    if errors:
        msg_lines = ["Fallaron algunos imports en genesis_modular (modo DEV):"]
        for k, v in errors.items():
            msg_lines.append(f" - {k}: {v}")
        full_msg = "\n".join(msg_lines)
        # En modo DEV queremos visibilidad explícita
        raise ImportError(full_msg)
    logger.debug("Comprobación DEV: todos los imports funcionaron correctamente.")


# Ejecutar la comprobación si la variable de entorno GENESIS_DEV está activada (solo para desarrollo)
if os.environ.get("GENESIS_DEV", "") in ("1", "true", "True"):
    _check_imports_dev()

# =============================================================================