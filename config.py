#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# dvt/config.py — Configuración Global del Pipeline DVT
# ==============================================================================
# Define y exporta variables de configuración clave para todo el paquete:
# - Logger global para mensajes consistentes.
# - Semilla de reproducibilidad.
# - Rutas de datos y del proyecto.
# - Flags de entorno para funcionalidades opcionales como JAX.
# ==============================================================================

import os
import sys
import logging
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# 1. Logger Global del Paquete
# ---------------------------------------------------------------------
# Se configura un único logger para ser usado en todo el proyecto.
logger = logging.getLogger("DVT")
logger.setLevel(logging.INFO)

# Evita duplicar handlers si el módulo se recarga
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------------------------------------------------------------------
# 2. Semilla para Reproducibilidad (CORRECCIÓN CLAVE)
# ---------------------------------------------------------------------
# La variable SEED se define aquí para que otros módulos puedan importarla.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
logger.info(f"Seed fijo: {SEED}")

# ---------------------------------------------------------------------
# 3. Rutas Base del Proyecto
# ---------------------------------------------------------------------
# Ruta base del proyecto (la carpeta que contiene el paquete 'genesis_modular')
BASE_PATH = Path(__file__).resolve().parent.parent.parent

# Ruta de datos, configurable con la variable de entorno 'DVT_DATA_DIR'
env_data_path = os.getenv("DVT_DATA_DIR")
if env_data_path:
    DATA_PATH = Path(env_data_path).expanduser().resolve()
else:
    # Por defecto, se asume una carpeta 'data' en la raíz del proyecto
    DATA_PATH = BASE_PATH / "data"

# Se crea el directorio de datos si no existe para evitar errores
if not DATA_PATH.exists():
    logger.warning(f"Directorio de datos no encontrado en '{DATA_PATH}'. Creando carpeta vacía.")
    DATA_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 4. Flags de Entorno
# ---------------------------------------------------------------------
# Permite activar JAX estableciendo la variable de entorno USE_JAX=1
USE_JAX = bool(int(os.getenv("USE_JAX", "0")))

if USE_JAX:
    # Desactiva la pre-asignación de memoria de JAX, que puede dar problemas
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import jax
        import jax.numpy as jnp
        logger.info(f"JAX v{jax.__version__} activado exitosamente. Usando jax.numpy.")
    except ImportError:
        logger.warning("USE_JAX=1 pero la librería JAX no está instalada. Se usará NumPy.")
        USE_JAX = False

# ---------------------------------------------------------------------
# 5. API Pública del Módulo
# ---------------------------------------------------------------------
# Define explícitamente qué variables se deben exportar de este módulo.
__all__ = [
    "logger",
    "SEED",
    "BASE_PATH",
    "DATA_PATH",
    "USE_JAX"
]