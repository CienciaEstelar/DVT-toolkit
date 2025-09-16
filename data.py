# dvt/data.py (VERSIÓN FINAL ROBUSTA)
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from astropy.io import ascii
import logging

from .config import DATA_PATH, logger

# =============================================================================
# VALIDACIÓN DE DATOS
# =============================================================================
def _validate_redshifts(z: np.ndarray, dataset_name: str) -> bool:
    """Valida que los redshifts sean físicamente válidos."""
    if np.any(z < 0):
        logger.error(f"{dataset_name}: redshifts negativos detectados")
        return False
    return True

def _validate_covariance(matrix: np.ndarray, n_points: int) -> bool:
    """Valida matrices de covarianza."""
    if matrix.shape != (n_points, n_points):
        logger.error(f"Matriz cov debe ser cuadrada ({n_points}x{n_points})")
        return False
    return True

# =============================================================================
# LECTORES DE DATOS (AJUSTADOS A TU ESTRUCTURA)
# =============================================================================
def _load_gw() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Carga datos de GW (puedes mantener tu implementación original si es necesario)"""
    return None

def _load_cmb() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Carga datos CMB desde tu archivo específico"""
    try:
        cmb_file = DATA_PATH / "5_COSMOLOGY/COM_PowerSpect_CMB-TT-full_R3.01.txt"
        if cmb_file.exists():
            tbl = ascii.read(cmb_file)
            return (
                tbl["l"].data.astype(float),
                tbl["Dl"].data.astype(float),
                tbl["-dDl"].data.astype(float),
                tbl["+dDl"].data.astype(float)
            )
    except Exception as e:
        logger.error(f"Error cargando CMB: {str(e)}")
    return None

def _load_bao() -> Optional[pd.DataFrame]:
    """Carga datos BAO desde tu CSV específico"""
    try:
        bao_file = DATA_PATH / "bao_data.csv"
        if bao_file.exists():
            df = pd.read_csv(bao_file, comment='#')
            required_cols = {"z_eff", "type", "value", "error"}
            if required_cols.issubset(df.columns):
                logger.info(f"✔ Datos BAO cargados: {len(df)} puntos")
                return df
    except Exception as e:
        logger.error(f"Error leyendo BAO: {str(e)}")
    return None

def _load_sn() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Carga datos de supernovas Pantheon+ con manejo robusto de headers y covarianza"""
    try:
        # Rutas a los archivos
        dat_file = DATA_PATH / "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
        cov_file = DATA_PATH / "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"
        
        # Leer datos de supernovas (ignorando comentarios y headers no numéricos)
        with open(dat_file) as f:
            sn_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Extraer redshifts y magnitudes distancia de forma segura
        z, mu = [], []
        for line in sn_lines:
            parts = line.split()
            try:
                z_val = float(parts[1])   # columna redshift
                mu_val = float(parts[4])  # columna distancia
                z.append(z_val)
                mu.append(mu_val)
            except (ValueError, IndexError):
                # Línea no numérica o incompleta → se ignora
                continue

        z = np.array(z)
        mu = np.array(mu)
        n = len(z)
        
        if n == 0:
            logger.error("No se cargaron datos SN válidos")
            return None
        
        # Leer matriz de covarianza (manejo preciso del formato)
        with open(cov_file) as f:
            cov_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # El primer elemento es el tamaño (1701), los siguientes son los valores
        if int(cov_lines[0]) != n:
            logger.error(f"Tamaño de matriz ({cov_lines[0]}) no coincide con datos ({n})")
            return None
            
        # Crear matriz de covarianza (ignorando el primer elemento que es el tamaño)
        cov_values = np.array([float(x) for x in cov_lines[1:n*n+1]])
        cov_matrix = cov_values.reshape(n, n)
        
        # Validaciones finales
        if not _validate_redshifts(z, "SN") or not _validate_covariance(cov_matrix, n):
            return None
            
        logger.info(f"✔ Datos SN cargados: {n} eventos (matriz {n}x{n})")
        return z, mu, cov_matrix
        
    except Exception as e:
        logger.error(f"Error cargando SN: {str(e)}", exc_info=True)
        return None

# =============================================================================
# API PÚBLICA
# =============================================================================
def load_data(required_datasets=("cmb", "sn")) -> Dict[str, Any]:
    """Carga datasets con validación de requeridos."""
    datasets = {
        "gw": _load_gw(),
        "cmb": _load_cmb(),
        "bao": _load_bao(),
        "sn": _load_sn()
    }

    # Validación de datasets requeridos
    missing = [name for name in required_datasets if datasets.get(name) is None]
    if missing:
        logger.critical(f"Datasets obligatorios faltantes: {', '.join(missing)}")
        sys.exit(1)

    # Verificar datasets obligatorios/advertencias
    if datasets.get("sn") is None:
        logger.critical("Se requieren datos de supernovas (SN) para el análisis")
        sys.exit(1)

    if datasets.get("cmb") is None:
        logger.warning("Advertencia: No se encontraron datos CMB")

    return {k: v for k, v in datasets.items() if v is not None}

__all__ = ["load_data"]
