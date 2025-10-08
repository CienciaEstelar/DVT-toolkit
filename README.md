# 🌌 Dynamic Vacuum Toolkit (DVT)

**Dynamic Vacuum Toolkit (DVT)** es un *framework* modular de cosmología escalar-tensor diseñado para el análisis de **energía oscura dinámica**, integrando física teórica, integración numérica e inferencia bayesiana de alto nivel.  
Su núcleo combina **derivación simbólica (SymPy)**, **resolución numérica (SciPy)**, y **aprendizaje automático (Procesos Gaussianos - GP)** bajo un pipeline reproducible y completamente automatizado.

---

## 🧠 Filosofía General

El DVT sigue una arquitectura basada en la secuencia:

> **Code → Solve → Learn → Infer**

Esta filosofía asegura trazabilidad total desde el Lagrangiano simbólico hasta la inferencia estadística sobre datos cosmológicos.

| Etapa | Módulo(s) | Descripción |
|:------|:-----------|:-------------|
| **Code** | `symbolic.py` | Deriva las ecuaciones de movimiento desde el Lagrangiano escalar-tensor. |
| **Solve** | `solver.py`, `cosmology.py` | Integra las EDOs para obtener la expansión cósmica \( H(z) \). |
| **Learn** | `potential.py` | Reconstruye el potencial escalar \( V(\phi) \) usando un Proceso Gaussiano. |
| **Infer** | `data.py`, `likelihood.py`, `mcmc.py` | Aplica inferencia bayesiana (MCMC) sobre observaciones cosmológicas. |
| **Run** | `run_pipeline.py` | Orquesta toda la cadena, desde la carga de datos hasta la generación de resultados finales. |

---

## 🧩 Estructura del Proyecto

| Archivo | Descripción Principal | Conexiones |
|:---------|:----------------------|:------------|
| **`__init__.py`** | Punto de entrada del paquete `genesis_modular`. Implementa *lazy imports*, define `__version__`, y expone las funciones públicas (`load_data`, `PotentialGP`, `DVT_MCMC`). | Conecta todos los módulos principales (`config`, `solver`, `mcmc`, etc.). |
| **`config.py`** | Configuración global del toolkit: logger central, semilla de reproducibilidad (`SEED = 42`), rutas base (`BASE_PATH`, `DATA_PATH`) y *flags* de entorno (`USE_JAX`). | Importado por todos los módulos para asegurar comportamiento determinista. |
| **`symbolic.py`** | Núcleo teórico del DVT. Deriva las ecuaciones de campo (ddot(Φ), ddot(a)) usando `sympy.euler_equations`. Genera versiones numéricas rápidas mediante *lambdify* y guarda caché con `cloudpickle`. | Exporta `phi_ddot_func` y `a_ddot_func` hacia `solver.py`. |
| **`potential.py`** | Clase `PotentialGP`: reconstrucción no paramétrica de \( V(\phi) \) mediante Proceso Gaussiano (`scikit-learn`). Incluye validación de hiperparámetros y serialización segura. | Utilizado por `mcmc.py` y `likelihood.py` para actualizar el potencial y sus derivadas. |
| **`solver.py`** | “Motor numérico” que integra las EDOs cosmológicas acopladas (Φ, a). Implementa seguridad física (`SafetyLimits`) y manejo de errores silencioso para estabilidad del MCMC. | Dependencia crítica de `likelihood.py`. Usa funciones de `symbolic.py`. |
| **`data.py`** | Entrada de datos observacionales (SN, CMB, BAO, GW). Valida *redshifts*, matrices de covarianza y consistencia de formatos. | Llamado por `run_pipeline.py` y `mcmc.py`. |
| **`cosmology.py`** | Clase `CosmoHelper`: cálculo de distancias cosmológicas comóviles, de luminosidad y diámetro angular mediante interpolación PCHIP. | Usado por `likelihood.py` para transformar \( H(z) \) en observables físicos. |
| **`likelihood.py`** | Núcleo de inferencia: clase `Likelihood`. Calcula la *log-verosimilitud* (-0.5 × χ²_total) y compara modelos con datos cosmológicos. | Base funcional del *sampler* MCMC (`mcmc.py`). |
| **`mcmc.py`** | Clase `DVT_MCMC`: implementa un *sampler* MCMC (usando `emcee`) para explorar el espacio de parámetros. Incluye paralelización, diagnóstico de convergencia y guardado seguro de resultados. | Depende de `likelihood.py`, `potential.py` y `data.py`. |
| **`run_pipeline.py`** | Script ejecutable principal (CLI). Configura, ejecuta y monitorea todo el pipeline, generando salidas gráficas, estadísticas y archivos finales. | Coordina todos los módulos (`data`, `mcmc`, `solver`, `cosmology`). |

---

## ⚙️ Comandos CLI

El archivo `run_pipeline.py` puede ejecutarse directamente desde la terminal usando **argumentos personalizables**.

```bash
python run_pipeline.py <ARGUMENTOS>
````

### Argumentos Disponibles

| Comando         | Tipo    | Por Defecto | Descripción                                                                                       |
| :-------------- | :------ | :---------- | :------------------------------------------------------------------------------------------------ |
| `--gp`          | `str`   | *None*      | **OBLIGATORIO:** ruta al modelo GP entrenado (`.pkl`). Contiene la reconstrucción de ( V(\phi) ). |
| `--walkers`     | `int`   | `100`       | Número de *walkers* (cadenas) en el muestreo MCMC.                                                |
| `--steps`       | `int`   | `10000`     | Iteraciones por cada cadena.                                                                      |
| `--burnin-frac` | `float` | `0.3`       | Fracción inicial de la cadena descartada como *burn-in*.                                          |
| `--thin`        | `int`   | `15`        | Factor de adelgazamiento (reduce correlación).                                                    |
| `--seed`        | `int`   | `42`        | Semilla de reproducibilidad.                                                                      |
| `--pool`        | `int`   | `0`         | Número de núcleos CPU usados. Si es 0, no hay paralelización.                                     |
| `--outdir`      | `str`   | *auto*      | Carpeta donde se guardarán resultados, gráficas y logs.                                           |
| `--z-max`       | `float` | `1100.0`    | Redshift máximo a integrar.                                                                       |
| `--phi-max`     | `float` | `1.5`       | Rango máximo del campo escalar Φ.                                                                 |

---

## 💻 Ejemplo de Ejecución Completa

```bash
python run_pipeline.py \
    --gp "models/best_gp_model.pkl" \
    --walkers 150 \
    --steps 50000 \
    --thin 20 \
    --pool 8 \
    --outdir "results/final_run_2025"
```

### 🔍 Desglose Interno del Pipeline

1. **Configuración de salida** → `_configure_output_dir()`
2. **Carga de datos** → `load_data()`
3. **Carga del modelo GP** → `_load_gp_model(args.gp)`
4. **Ejecución del MCMC** → `DVT_MCMC.run()`
5. **Cálculo de burn-in** → `_compute_burnin()`
6. **Generación de resultados** → `_generate_plots()` y `_save_results()`

---

## 🧮 Características Clave

* Derivación simbólica automática del Lagrangiano (SymPy)
* Resolución numérica robusta con validación de estabilidad
* Reconstrucción GP no paramétrica del potencial escalar
* Muestreo MCMC multiproceso con diagnóstico de convergencia
* Sistema completo de logging y manejo de errores
* Resultados reproducibles con trazabilidad total

---

## 📦 Requisitos

```bash
Python >= 3.9
pip install -r requirements.txt
```

**Dependencias principales:**

* `numpy`, `scipy`, `sympy`, `matplotlib`
* `scikit-learn`, `emcee`, `cloudpickle`
* `optuna` *(opcional para tuning de hiperparámetros)*

---

## 📁 Salidas del Pipeline

El toolkit genera automáticamente una carpeta `results/` con:

* 🧩 `chains/` → Cadenas MCMC completas
* 📊 `plots/` → Corner plots, evolución de parámetros, H(z), etc.
* 📜 `summary.txt` → Estadísticas y convergencia
* 🧠 `model_state.pkl` → Estado del GP y parámetros cosmológicos
* 🧾 `log.txt` → Registro detallado de ejecución

---

## 🛰️ Contribuciones y Créditos

Proyecto desarrollado por **Juan de Dios Galaz (CienciaEstelar)**
Estudiante de Ingeniería en Ejecución en Minas (USACH) y divulgador científico.
Inspirado por la necesidad de unir **gravedad cuántica, energía oscura y modelos falsables** bajo una arquitectura reproducible.

---

## 📖 Cita Recomendada

> Galaz, J. (2025). *Dynamic Vacuum Toolkit (DVT): A Symbolic–Numerical Framework for Dynamic Dark Energy Inference.*
> GitHub Repository: [https://github.com/CienciaEstelar/DVT-toolkit](https://github.com/CienciaEstelar/DVT-toolkit)

---

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=CienciaEstelar&repo=DVT-toolkit&theme=tokyonight" alt="DVT Toolkit Stats"/>
</p>
