# dvt/cosmology.py
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.interpolate import PchipInterpolator
from astropy import units as u
import logging

logger = logging.getLogger('DVT_COSMO')

class CosmoHelper:
    """Calculador de distancias cosmológicas con interpolación optimizada."""
    def __init__(self, H0: float):
        self.H0 = H0 * u.km / u.s / u.Mpc
        self._c = 299_792.458 * u.km / u.s

    # <<< INICIO DE LA FUNCIÓN ACTUALIZADA >>>
    # -----------------------------------------------------------
    def _create_interpolator(self, z_data: np.ndarray, H_data: np.ndarray) -> Optional[PchipInterpolator]:
        """Crea interpolador PCHIP con validación robusta."""
        # Validación exhaustiva de inputs
        if not (isinstance(z_data, np.ndarray) and isinstance(H_data, np.ndarray)):
            logger.error("Inputs deben ser numpy arrays")
            return None

        if z_data.size != H_data.size or z_data.size < 1:
            logger.error(f"Tamaños inconsistentes: z({z_data.size}), H({H_data.size})")
            return None

        if not (np.all(np.isfinite(z_data)) and np.all(np.isfinite(H_data))):
            logger.error("NaN/inf encontrados en z_data o H_data")
            return None

        # Limpieza automática de datos (orden + únicos)
        sort_idx = np.argsort(z_data)
        z_data = z_data[sort_idx]
        H_data = H_data[sort_idx]

        # Eliminar duplicados con tolerancia numérica
        unique_mask = np.concatenate(([True], np.diff(z_data) > 1e-10))
        z_data = z_data[unique_mask]
        H_data = H_data[unique_mask]

        if z_data.size < 1:
            logger.error("No hay suficientes puntos únicos tras limpieza")
            return None

        # Extensión a z=0 con H0 (si no está ya presente)
        if not np.isclose(z_data[0], 0.0, atol=1e-12):
            z_ext = np.concatenate(([0.0], z_data))
            H_ext = np.concatenate(([self.H0.to(u.km / u.s / u.Mpc).value], H_data))
        else:
            z_ext = z_data.copy()
            H_ext = H_data.copy()
            # For consistency, replace the H at z=0 with the provided H0 value
            H_ext[0] = self.H0.to(u.km / u.s / u.Mpc).value

        return PchipInterpolator(z_ext, H_ext, extrapolate=False)

    def comoving_distance(self, z: float, z_grid: np.ndarray, H_grid: np.ndarray) -> float:
        """Distancia comóvil con protección numérica robusta."""
        if z < 0 or not np.isfinite(z):
            return 0.0

        interp = self._create_interpolator(z_grid, H_grid)
        if interp is None:
            return np.nan

        try:
            z_max_interp = interp.x.max()
            z_eval_max = min(z, z_max_interp)

            # Generar puntos de evaluación evitando duplicados en el límite
            if np.isclose(z_eval_max, z_max_interp, atol=1e-8):
                z_eval = np.linspace(0.0, z_eval_max, 199)  # Número impar para evitar z=z_max
                z_eval = np.append(z_eval, z_max_interp)   # Añadir manualmente el límite
            else:
                z_eval = np.linspace(0.0, z_eval_max, 200)

            # Asegurar orden estricto de los puntos
            z_eval = np.unique(z_eval)
            if not np.all(np.diff(z_eval) > 0):
                raise RuntimeError("Puntos de evaluación no ordenados")

            H_eval = interp(z_eval)
            return float(np.trapz(self._c.to(u.km / u.s).value / H_eval, z_eval))
        except Exception as e:
            logger.error(f"Error en comoving_distance: {str(e)}", exc_info=True)
            return np.nan

    def luminosity_distance(self, z: float, z_grid: np.ndarray, H_grid: np.ndarray) -> float:
        """Distancia de luminosidad en Mpc."""
        Dc = self.comoving_distance(z, z_grid, H_grid)
        return (1.0 + z) * Dc if np.isfinite(Dc) else np.nan

    def angular_diameter_distance(self, z: float, z_grid: np.ndarray, H_grid: np.ndarray) -> float:
        """Distancia angular en Mpc."""
        Dc = self.comoving_distance(z, z_grid, H_grid)
        return Dc / (1.0 + z) if np.isfinite(Dc) else np.nan

__all__ = ['CosmoHelper']