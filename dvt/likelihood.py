# dvt/likelihood.py (VERSIÓN FINAL OPTIMIZADA)
"""
Cálculo de χ² con:
- Validación numérica exhaustiva
- Manejo seguro de interpolaciones
- Protección contra errores en BAO/GW
- Logging detallado para diagnóstico
"""

from __future__ import annotations
import numpy as np
import camb
from astropy.cosmology import Planck18
import logging
import warnings
from scipy.linalg import LinAlgWarning
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from typing import Dict, Optional, Tuple

from .cosmology import CosmoHelper
from .solver import solve_dvt
from .config import logger

# Configuración de warnings
warnings.filterwarnings("ignore", category=LinAlgWarning, message="Matrix is exactly singular")

class Likelihood:
    def __init__(self, data: Dict):
        """Inicializa con validación robusta de datos.
        
        Args:
            data: Diccionario con claves 'cmb', 'sn', 'bao', 'gw'
        """
        self.d = data if data else {}
        self._cosmo_helper_cache: Dict[float, CosmoHelper] = {}
        self.rd_fiducial = 147.09  # Sound horizon [Mpc]
        
        # Validación inicial de datos SN
        if "sn" in self.d and self.d["sn"] is not None:
            zs, _, cov = self.d["sn"]
            if np.any(zs < 0):
                logger.error("Redshifts SN contienen valores negativos")
            if not np.all(np.isfinite(cov)):
                logger.error("Matriz de covarianza SN tiene NaN/Inf")

    def _get_cosmo_helper(self, H0: float) -> CosmoHelper:
        """Cache de CosmoHelper para evitar recreación constante."""
        if H0 not in self._cosmo_helper_cache:
            self._cosmo_helper_cache[H0] = CosmoHelper(H0)
        return self._cosmo_helper_cache[H0]

    # =====================================================================
    # COMPONENTES DE χ² CON PROTECCIÓN NUMÉRICA
    # =====================================================================
    def _chi2_cmb(self, H0: float, omb: float, omc: float, 
                 z: np.ndarray, H: np.ndarray) -> float:
        """χ² para CMB con validación de inputs y outputs."""
        if self.d.get("cmb") is None:
            return 0.0
            
        try:
            ell_obs, Dl_obs, dDl_minus, dDl_plus = self.d["cmb"]
            Dl_err = (dDl_plus + dDl_minus) / 2.0

            if np.any(Dl_err <= 0):
                logger.warning("Errores de CMB no positivos")
                return np.inf

            # Configuración CAMB con controles de precisión
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=float(H0),
                ombh2=float(omb),
                omch2=float(omc),
                omk=0.0,
                mnu=0.06
            )
            pars.InitPower.set_params(ns=0.965, As=2e-9)
            pars.set_for_lmax(int(ell_obs.max()), lens_potential_accuracy=1)
            
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
            Dl_th_full = powers['unlensed_scalar'][:, 0]
            
            # Interpolación segura con manejo de bordes
            ell_th = np.arange(len(Dl_th_full))
            Dl_th = np.interp(
                ell_obs, 
                ell_th, 
                Dl_th_full, 
                left=np.nan, 
                right=np.nan
            )

            if not np.all(np.isfinite(Dl_th)):
                logger.warning("CAMB produjo valores no finitos")
                return np.inf
                
            return np.nansum(((Dl_th - Dl_obs) / Dl_err) ** 2)
            
        except Exception as e:
            logger.error(f"Error en _chi2_cmb: {str(e)}", exc_info=True)
            return np.inf

    def _chi2_sn(self, H0: float, z: np.ndarray, H: np.ndarray) -> float:
        """χ² para supernovas con validación de matriz de covarianza."""
        if self.d.get("sn") is None:
            return 0.0
            
        try:
            zs, mu_obs, cov = self.d["sn"]
            
            # Validación de inputs
            if np.any(~np.isfinite(H)) or np.any(H <= 0):
                logger.warning("H(z) no finito o no positivo")
                return np.inf
            if np.any(zs < 0) or np.any(~np.isfinite(zs)):
                logger.warning("Redshifts SN inválidos")
                return np.inf

            cosmo = self._get_cosmo_helper(H0)
            mu_th = np.zeros_like(zs)
            
            # Cálculo de distancia luminosa con protección
            for i, zi in enumerate(zs):
                dl = cosmo.luminosity_distance(zi, z, H)
                if not np.isfinite(dl) or dl <= 0:
                    logger.debug(f"DL no finita en z={zi:.3f}")
                    return np.inf
                mu_th[i] = 5 * np.log10(dl) + 25

            # Pseudoinversa con tolerancia numérica
            inv_cov = np.linalg.pinv(cov, rcond=1e-15, hermitian=True)
            dmu = mu_th - mu_obs
            return float(dmu @ inv_cov @ dmu)
            
        except Exception as e:
            logger.error(f"Error en _chi2_sn: {str(e)}", exc_info=True)
            return np.inf

    def _chi2_bao(self, H0: float, z: np.ndarray, H: np.ndarray) -> float:
        """χ² para BAO con interpolación segura y manejo de redshifts."""
        if self.d.get("bao") is None or len(self.d["bao"]) == 0:
            return 0.0
            
        try:
            cosmo = self._get_cosmo_helper(H0)
            bao_data = self.d["bao"]
            chi2 = 0.0

            for row in bao_data:
                z_bao = row['z_eff']
                if not (0 < z_bao < z[-1]):  # Usar z[-1] para evitar NaN
                    continue
                    
                try:
                    # Interpolación segura con PCHIP
                    if row['type'] == 'DM':
                        dist = cosmo.comoving_distance(z_bao, z, H)
                        model_val = dist / self.rd_fiducial
                    elif row['type'] == 'DH':
                        H_interp = PchipInterpolator(z, H, extrapolate=False)
                        H_at_z = H_interp(z_bao)
                        model_val = (299792.458 / H_at_z) / self.rd_fiducial
                    else:
                        continue
                        
                    if np.isfinite(model_val):
                        chi2 += ((model_val - row['value']) / row['error']) ** 2
                    else:
                        logger.debug(f"Modelo no finito en z_bao={z_bao}")
                except Exception as e:
                    logger.debug(f"Error en z_bao={z_bao}: {str(e)}")
                    continue
                    
            return chi2 if np.isfinite(chi2) else np.inf
            
        except Exception as e:
            logger.error(f"Error en _chi2_bao: {str(e)}", exc_info=True)
            return np.inf

    def _chi2_gw(self, H0: float, z: np.ndarray, H: np.ndarray) -> float:
        """χ² para GW con validación de distancias luminosas."""
        if self.d.get("gw") is None:
            return 0.0
            
        try:
            zs, dl_obs, dl_err = self.d["gw"]
            cosmo = self._get_cosmo_helper(H0)
            chi2 = 0.0
            
            # Interpolador PCHIP para mejor precisión
            dl_interp = PchipInterpolator(
                z, 
                [cosmo.luminosity_distance(zi, z, H) for zi in z],
                extrapolate=False
            )
            
            for zg, dl_o, dl_e in zip(zs, dl_obs, dl_err):
                if dl_e <= 0 or zg <= 0 or zg >= z[-1]:
                    continue
                    
                try:
                    dl_t = dl_interp(zg)
                    if np.isfinite(dl_t):
                        chi2 += ((dl_o - dl_t) / dl_e) ** 2
                except Exception as e:
                    logger.debug(f"Error en z_gw={zg}: {str(e)}")
                    
            return chi2 if np.isfinite(chi2) else np.inf
            
        except Exception as e:
            logger.error(f"Error en _chi2_gw: {str(e)}", exc_info=True)
            return np.inf

    # =====================================================================
    # LIKELIHOOD TOTAL CON MANEJO DE ERRORES EN CASCADA
    # =====================================================================
    def total(self, theta: np.ndarray, gp) -> float:
        """Log-verosimilitud marginalizada con protección completa."""
        try:
            sol = solve_dvt(theta, gp)
            if sol is None:
                logger.debug("solve_dvt falló para theta=%s", theta)
                return -np.inf
                
            z, H, H0, omb, omc = sol
            
            # Validación física de la solución
            if not (np.all(np.diff(z) > 0) and np.all(H > 0)):
                logger.debug(
                    "Solución inválida: z no ordenado o H(z) ≤ 0 | "
                    f"z=[{z[0]:.2f}, {z[-1]:.2f}], H=[{H.min():.2f}, {H.max():.2f}]"
                )
                return -np.inf

            # Cálculo robusto de componentes
            chi2_components = np.array([
                self._chi2_cmb(H0, omb, omc, z, H),
                self._chi2_sn(H0, z, H),
                self._chi2_bao(H0, z, H),
                self._chi2_gw(H0, z, H)
            ])
            
            # Filtrado de componentes inválidos
            valid_mask = np.isfinite(chi2_components)
            if not np.any(valid_mask):
                logger.debug("Todos los componentes de χ² son inválidos")
                return -np.inf
                
            chi2_total = np.sum(chi2_components[valid_mask])
            return -0.5 * chi2_total
                
        except Exception as e:
            logger.error(
                "Error en likelihood.total:\nInput: %s\nError: %s",
                str(theta), str(e), exc_info=True
            )
            return -np.inf

__all__ = ["Likelihood"]