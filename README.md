# 🌀 Dynamic Vacuum Toolkit (DVT)

### Framework Cosmológico Escalar-Tensor y Análisis Bayesiano de Energía Oscura Dinámica

El **Dynamic Vacuum Toolkit (DVT)** es un *framework* modular y robusto para el estudio cosmológico de modelos de **energía oscura dinámica** (tipo escalar-tensor).  
Combina derivación simbólica, integración numérica e inferencia estadística mediante **Procesos Gaussianos (GP)** y **MCMC** (Cadena de Markov Monte Carlo).  

Su arquitectura garantiza **trazabilidad completa**, **reproducibilidad**, y **control de errores a nivel de módulo**, asegurando la estabilidad en simulaciones prolongadas.

---

## ⚙️ Filosofía y Flujo de Ejecución

El DVT sigue el principio **Code → Solve → Learn → Infer → Run**, donde cada módulo cumple una etapa específica del flujo cosmológico:

1. **Code (`symbolic.py`)** → Deriva las ecuaciones de movimiento a partir del lagrangiano escalar-tensor.  
2. **Solve (`solver.py`, `cosmology.py`)** → Integra las EDOs para obtener la expansión cósmica `H(z)`.  
3. **Learn (`potential.py`)** → Reconstruye el potencial escalar `V(φ)` mediante un Proceso Gaussiano (GP).  
4. **Infer (`data.py`, `likelihood.py`, `mcmc.py`)** → Combina datos observacionales con el modelo y realiza inferencia bayesiana (`χ²`, MCMC).  
5. **Run (`run_pipeline.py`)** → Orquesta el *pipeline* completo de principio a fin.

---

## 🧩 Arquitectura del Proyecto

| Módulo | Descripción | Conexiones |
|--------|--------------|------------|
| **`__init__.py`** | Punto de entrada y API pública del paquete `genesis_modular`. Implementa *lazy imports* para optimización y define el mapa global de clases y funciones (`load_data`, `PotentialGP`, `DVT_MCMC`). | Conecta con `config`, `data`, `potential`, `solver`, `cosmology`, `likelihood`, `mcmc`. |
| **`config.py`** | Módulo central de configuración global. Define el *logger* (`logging.getLogger("DVT")`), la semilla de reproducibilidad (`SEED=42`), rutas base y *flags* (`USE_JAX`). | Importado por todos los módulos para mantener consistencia global. |
| **`symbolic.py`** | Núcleo teórico: deriva las ecuaciones de campo (Φ̈, ä) mediante `sympy.euler_equations`. Incluye lambdificación y caché (`cloudpickle`). | Exporta `phi_ddot_func` y `a_ddot_func` a `solver.py`. Usa `config.py` para *logging* y caché. |
| **`potential.py`** | Contiene la clase `PotentialGP`, que reconstruye `V(φ)` con un Proceso Gaussiano (GP). Incluye validación, serialización (`save/load`), y escalado físico del potencial. | Usado por `mcmc.py` (actualización GP) y `likelihood.py` (evaluación de V y V'). |
| **`solver.py`** | Motor numérico: integra las EDOs cosmológicas acopladas para obtener `H(z)`. Incluye control de singularidades (`SafetyLimits`) y caché LRU para rendimiento. | Dependencia principal de `likelihood.py`. Usa funciones de `symbolic.py`. |
| **`data.py`** | Carga y valida datos observacionales (SN, CMB, BAO, GW). Implementa validación de *redshifts* y covarianzas. | Llamado por `run_pipeline.py` y `mcmc.py`. Alimenta la clase `Likelihood`. |
| **`cosmology.py`** | Clase `CosmoHelper` para cálculos de distancias cosmológicas (`∫ dz/H(z)`), luminosidad y diámetro angular. Optimizado con interpolación PCHIP. | Utilizado por `likelihood.py` para comparar modelo con datos observacionales. |
| **`likelihood.py`** | Contiene la clase `Likelihood`, que calcula la *log-verosimilitud total* (`−0.5 χ²_total`) a partir de las comparaciones entre modelo y datos. | Núcleo del *sampler* `mcmc.py`. Usa `solver.py` y `cosmology.py`. |
| **`mcmc.py`** | Implementa la clase `DVT_MCMC` (basada en `emcee`). Explora el espacio de parámetros cosmológicos con multiprocesamiento, control de convergencia y guardado seguro. | Depende de `likelihood.py`, `potential.py`, y `data.py`. |
| **`run_pipeline.py`** | Script ejecutable principal (`__main__`). Define CLI, parámetros del MCMC y genera resultados gráficos (`corner plots`, `H(z)`, potenciales). | Coordina todos los módulos para ejecutar la inferencia completa. |

---

## 🧠 Capacidades Principales

- 🔹 **Derivación simbólica exacta** (Lagrangiano escalar-tensor con `sympy`).
- 🔹 **Integración numérica estable** con control de errores físicos.
- 🔹 **Reconstrucción no paramétrica** del potencial escalar mediante GP.
- 🔹 **Inferencia bayesiana MCMC** con diagnóstico de convergencia (`τ`, acceptance rate).
- 🔹 **Gestión inteligente de cachés** para evitar recálculos y acelerar la ejecución.
- 🔹 **Pipeline reproducible** de extremo a extremo: derivación → solución → inferencia.

---

## 📦 Dependencias Principales

- `sympy` — Derivación simbólica y euler equations.  
- `numpy`, `scipy` — Cálculo numérico y EDOs.  
- `scikit-learn` — Procesos Gaussianos (GP).  
- `emcee` — Sampler MCMC.  
- `matplotlib` / `corner` — Visualización de resultados.  
- `cloudpickle` — Caché simbólica y serialización.

---

## ▶️ Ejecución Rápida

### 1. Instalación
```bash
pip install -r requirements.txt
````

### 2. Correr el Pipeline Completo

```bash
python run_pipeline.py
```

### 3. Verificar Blindaje Teórico (RG)

```bash
python analysis/derivacion_vc.py
```

---

## 📊 Salidas del Sistema

Los resultados se almacenan automáticamente en la carpeta `output/`:

* 🧮 **Ecuaciones simbólicas** (`.tex`)
* 📈 **Gráficos cosmológicos** (`.png`, `.pdf`)
* 📊 **Resultados MCMC** (`chains.h5`, `corner_plots/`)

---

## 🧩 Proyecto y Autoría

**Autor:** Juan Galaz
**Framework:** Dynamic Vacuum Toolkit (DVT) — *Geometría Causal-Informacional (GCI)*
**Institución:** Universidad de Santiago de Chile (USACH)

---

## 📜 Licencia

Distribuido bajo licencia **MIT**.
Promueve la investigación abierta, reproducible y colaborativa en cosmología teórica.
