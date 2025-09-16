# genesis_modular/run_pipeline.py (VERSIÓN FINAL CON GRÁFICOS HQ)
"""
Pipeline principal DVT con:
- Manejo robusto de errores
- Paralelización segura
- Generación automática de reportes en alta calidad (PNG @ 1200 DPI y PDF)
"""

from __future__ import annotations
import argparse
import logging
import os
import warnings
from multiprocessing import freeze_support
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import sys
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# Configuración de warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar subtract")

# Módulos DVT (usando importaciones relativas correctas)
from .dvt.config import logger
from .dvt.data import load_data
from .dvt.mcmc import DVT_MCMC
from .dvt.potential import PotentialGP
from .dvt.solver import solve_dvt
from .dvt.cosmology import CosmoHelper
# Importaciones clave para la deserialización del modelo GP
from .train_gp import BayesianEnsembleGP, PolynomialKernel

# =============================================================================
# CONFIGURACIÓN Y UTILIDADES
# =============================================================================

def _load_train_gp_pickle(gp_path: str) -> PotentialGP:
    """Carga específicamente pickles de train_gp.py usando los parámetros correctos."""
    try:
        import pickle
        
        with open(gp_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info("✅ Archivo .pkl cargado exitosamente. Extrayendo parámetros...")

        if 'args' in data and isinstance(data['args'], dict):
            args = data['args']
            phi_min, phi_max, n_points, m_phi, extend_range = -5.0, 5.0, 1000, 0.5, 0.2
            logger.info(f"📦 Parámetros extraídos del .pkl: φ=[{phi_min},{phi_max}], n={n_points}, m_φ={m_phi}")
        else:
            logger.warning("No se encontraron 'args' en el .pkl, usando valores por defecto.")
            phi_min, phi_max, n_points, m_phi, extend_range = -5.0, 5.0, 1000, 0.5, 0.2
        
        physical_scale = 0.5 * (m_phi ** 2) * (phi_max ** 2)
        range_extension = extend_range * (phi_max - phi_min)
        effective_phi_min, effective_phi_max = phi_min - range_extension, phi_max + range_extension
        
        gp_instance = PotentialGP(
            n_anchors=n_points,
            anchor_span=(effective_phi_min, effective_phi_max),
            optimizer="fmin_l_bfgs_b",
            physical_scale=physical_scale,
            meta={"source": "train_gp_pickle", "original_args": args if 'args' in data else {}}
        )
        
        if 'gp' in data and hasattr(data['gp'], 'models'):
            try:
                best_model_dict = max(
                    (m for m in data['gp'].models if hasattr(m.get('gp'), 'log_marginal_likelihood_value_')),
                    key=lambda m: m['gp'].log_marginal_likelihood_value_,
                    default=None
                )

                if best_model_dict:
                    best_gp = best_model_dict['gp']
                    kernel_params = best_gp.kernel_.get_params()
                    logA = np.log(kernel_params.get('k1__k1__constant_value', 1.0))
                    logL = np.log(kernel_params.get('k1__k2__length_scale', 1.0))
                    gp_instance.update([logA, logL])
                    logger.info(f"✅ Hiperparámetros transferidos del mejor modelo: logA={logA:.3f}, logL={logL:.3f}")
                else:
                    logger.warning("⚠️ No se encontró un modelo válido con log_marginal_likelihood en el ensamble.")
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron transferir hiperparámetros: {e}")
        
        logger.info(f"🎯 PotentialGP reconstruido a partir de {gp_path}")
        return gp_instance
            
    except Exception as e:
        logger.error(f"❌ Fallo crítico en la carga especializada: {e}", exc_info=True)
        logger.warning("🔄 Usando GP de fallback como último recurso.")
        return PotentialGP(n_anchors=1000, anchor_span=(-6.0, 6.0))

def _configure_output_dir(outdir: Optional[str]) -> Path:
    """Crea directorio de salida con validación."""
    if outdir:
        dir_path = Path(outdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_path = Path(f"results/run_{timestamp}")
    
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Directorio de salida: {dir_path.resolve()}")
    return dir_path

def _load_gp_model(gp_path: Optional[str], phi_max: float) -> PotentialGP:
    """Carga o inicializa el modelo GP con manejo robusto de errores."""
    if gp_path and Path(gp_path).exists():
        return _load_train_gp_pickle(gp_path)
    
    logger.warning("⚠️ No se proporcionó ruta a GP o no existe. Usando GP por defecto.")
    gp_dir = Path("./dvt/gp_models")
    gp_dir.mkdir(exist_ok=True)
    gp = PotentialGP(n_anchors=31, anchor_span=(-phi_max, phi_max))
    gp.save(gp_dir / "gp_default.pkl")
    return gp

def _compute_burnin(sampler: DVT_MCMC, burnin_frac: float) -> int:
    """Calcula burn-in basado en autocorrelación o fracción fija."""
    try:
        tau = sampler.get_autocorr_time(tol=0.01)
        max_tau = int(np.nanmax(tau))
        burn = max(100, 3 * max_tau)
        logger.info(f"Autocorrelación máxima: {max_tau} → burn-in ajustado: {burn}")
    except Exception as e:
        chain_len = sampler.get_chain().shape[1]
        burn = int(burnin_frac * chain_len)
        logger.warning(f"Usando burn-in por fracción ({burnin_frac:.0%}): {burn} pasos (error en autocorr: {str(e)})")
    return burn

# =============================================================================
# GENERACIÓN DE RESULTADOS
# =============================================================================
def _generate_plots(
    outdir: Path,
    chain: np.ndarray,
    best_params: np.ndarray,
    gp: PotentialGP,
    phi_max: float,
    z_max: float
) -> None:
    """Genera todos los gráficos y artefactos de análisis en alta calidad."""
    logger.info("📊 Generando gráficos de resultados en alta calidad (PNG @ 1200 DPI y PDF)...")
    param_labels = ["H0", "Ω_bh²", "Ω_ch²", "φ_0", "φ̇_0", "ξ", "logA", "logL"]
    try:
        fig = corner.corner(
            chain[:, :6],
            labels=param_labels[:6],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 10}
        )
        # --- MODIFICACIÓN: Guardar en PNG (1200 DPI) y PDF ---
        fig.savefig(outdir / "corner_plot.png", dpi=1200, bbox_inches="tight")
        fig.savefig(outdir / "corner_plot.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.error(f"No se pudo generar el corner plot: {e}")

    try:
        phi_grid = np.linspace(-phi_max * 1.5, phi_max * 1.5, 500)
        V_phi, V_std = gp.physical_potential(phi_grid, return_std=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(phi_grid, V_phi, 'b-', lw=2, label="Potencial Mediano")
        plt.fill_between(phi_grid, V_phi - V_std, V_phi + V_std, color='blue', alpha=0.2, label="Incertidumbre 1σ")
        plt.xlabel("φ [unidades reducidas]", fontsize=12)
        plt.ylabel("V(φ) [unidades físicas]", fontsize=12)
        plt.title("Potencial escalar reconstruido", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        # --- MODIFICACIÓN: Guardar en PNG (1200 DPI) y PDF ---
        plt.savefig(outdir / "potential.png", dpi=1200, bbox_inches="tight")
        plt.savefig(outdir / "potential.pdf", bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.error(f"No se pudo generar el gráfico del potencial: {e}")

    try:
        hz_solution = solve_dvt(best_params, gp, z_max=z_max)
        if hz_solution is not None:
            z, H, H0, ombh2, omch2 = hz_solution
            plt.figure(figsize=(10, 6))
            plt.plot(z, H, 'r-', lw=2)
            plt.xlabel("Redshift (z)", fontsize=12)
            plt.ylabel("H(z) [km/s/Mpc]", fontsize=12)
            plt.title(f"Evolución de Hubble\n$H_0$={H0:.1f}, $Ω_m$={(ombh2 + omch2)/(H0/100)**2:.3f}", fontsize=14)
            plt.grid(True, alpha=0.3)
            # --- MODIFICACIÓN: Guardar en PNG (1200 DPI) y PDF ---
            plt.savefig(outdir / "hubble_evolution.png", dpi=1200, bbox_inches="tight")
            plt.savefig(outdir / "hubble_evolution.pdf", bbox_inches="tight")
            plt.close()
            np.savetxt(outdir / "hubble_data.txt", np.column_stack((z, H)), header="z H(z)[km/s/Mpc]")
        else:
            logger.error("No se pudo calcular H(z) para los mejores parámetros.")
    except Exception as e:
        logger.error(f"No se pudo generar el gráfico de H(z): {e}")

def _save_results(
    outdir: Path,
    chain: np.ndarray,
    sampler: DVT_MCMC,
    gp: PotentialGP,
    args: argparse.Namespace
) -> None:
    """Guarda todos los resultados en formato estándar."""
    logger.info("💾 Guardando resultados numéricos...")
    best_params = np.median(chain, axis=0)
    
    np.save(outdir / "mcmc_chain_full.npy", sampler.get_chain())
    np.save(outdir / "mcmc_chain_flat.npy", chain)
    np.savetxt(outdir / "best_params.txt", best_params)
    
    gp.update(best_params[6:8])
    gp.save(outdir / "best_gp_model.pkl")
    
    param_names = ["H0", "ombh2", "omch2", "phi0", "phidot0", "xi", "logA", "logL"]
    summary = pd.DataFrame(chain, columns=param_names).describe(percentiles=[.16, .50, .84]).transpose()
    summary.to_csv(outdir / "mcmc_summary.csv")
    
    with open(outdir / "metadata.txt", "w") as f:
        f.write(f"# Metadata - DVT Pipeline\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Config: {vars(args)}\n")

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def run_pipeline(args: argparse.Namespace) -> Tuple[bool, Path]:
    """Ejecuta el flujo completo del pipeline DVT."""
    try:
        outdir = _configure_output_dir(args.outdir)
        
        logger.info("1. Cargando datos observacionales...")
        data = load_data()
        
        logger.info("2. Configurando modelo GP...")
        gp = _load_gp_model(args.gp, args.phi_max)
        
        logger.info(f"3. Iniciando MCMC con {args.walkers} walkers ({args.steps} pasos)...")
        
        sampler = DVT_MCMC(
            data=data,
            walkers=args.walkers,
            steps=args.steps,
            use_pool=(args.pool > 0),
            output_dir=outdir,
            seed=args.seed
        )
        sampler.run()
        
        logger.info("4. Procesando cadenas MCMC...")
        burn = _compute_burnin(sampler, args.burnin_frac)
        chain = sampler.get_chain(flat=True, discard=burn, thin=args.thin)
        best_params = np.median(chain, axis=0)
        
        logger.info("5. Actualizando GP y generando artefactos...")
        gp.update(best_params[6:8])
        
        _generate_plots(outdir, chain, best_params, gp, args.phi_max, args.z_max)
        _save_results(outdir, chain, sampler, gp, args)
        
        logger.info(f"🚀 Pipeline completado! Resultados en: {outdir.resolve()}")
        return True, outdir
        
    except Exception as e:
        logger.critical(f"❌ Fallo crítico en el pipeline: {str(e)}", exc_info=True)
        return False, Path(".")

def _cli() -> argparse.Namespace:
    """Configura el parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline DVT para análisis cosmológico con campos escalares",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gp", type=str, required=True, help="Ruta al modelo GP entrenado (.pkl)")
    parser.add_argument("--walkers", type=int, default=100, help="Número de walkers MCMC")
    parser.add_argument("--steps", type=int, default=10000, help="Pasos por walker")
    parser.add_argument("--burnin-frac", type=float, default=0.3, 
                       help="Fracción de burn-in si no se detecta autocorrelación")
    parser.add_argument("--thin", type=int, default=15, help="Factor de thinning.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--pool", type=int, default=0, help="Núcleos para paralelización (0 para desactivar).")
    parser.add_argument("--outdir", type=str, help="Directorio de salida.")
    parser.add_argument("--z-max", type=float, default=1100.0, help="Redshift máximo para integración")
    parser.add_argument("--phi-max", type=float, default=1.5, help="Rango máximo para reconstrucción de φ")
    return parser.parse_args()

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    freeze_support()
    args = _cli()
    np.random.seed(args.seed)
    
    success, results_dir = run_pipeline(args)
    if success:
        logger.info(f"🟢 Análisis completado. Resultados en: {results_dir.resolve()}")
    else:
        logger.error("🔴 El pipeline falló. Revisa los logs para más detalles.")
        sys.exit(1)