# 🌌 Geometría Causal-Informacional (GCI)  
### Framework de Cosmología Escalar-Tensor (DVT)

Este repositorio contiene el **Dynamic Vacuum Toolkit (DVT)**, enfocado en la implementación de modelos de **Energía Oscura Dinámica**.  
El proyecto principal es la **Geometría Causal-Informacional (GCI)**, que unifica principios de información, gravedad cuántica y cosmología.

---

## 🔎 Descripción del Proyecto

El script principal, `modelo_cosmologico.py`, es un **framework de cosmología** que realiza:

- **Derivación Simbólica:** Genera las Ecuaciones de Klein-Gordon y Friedmann modificadas a partir del Lagrangiano (usando *SymPy*).  
- **Calibración:** Fija la frecuencia de corte fundamental (νₐ) y la densidad de energía del vacío (ρₐ) con alta precisión.  
- **Simulación Numérica:** Integra las EDOs para la evolución cosmológica *a(t)* y *Φ(t)*.  
- **Predicciones Clave:** Calcula la masa del axión predicha (*mₐ ≈ 3.61 meV*).

---

## 📂 Estructura del Repositorio

| Carpeta | Contenido Principal | Propósito |
| :--- | :--- | :--- |
| `/` | Scripts de ejecución principal (`modelo_cosmologico.py`) | Ejecución central del modelo |
| `analysis/` | Scripts de verificación de consistencia | Blindaje teórico (e.g. `derivacion_vc.py` para Punto Fijo RG) |
| `docs/` | Manuscritos, apéndices y documentación | Referencia teórica |
| `output/` | Archivos generados | Ecuaciones (LaTeX), gráficos (.png, .pdf) y resultados de simulación |

---

## ▶️ Instrucciones de Reproducción

### 1️⃣ Instalación de Dependencias

Se requiere **Python 3.9+** y las librerías listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
````

---

### 2️⃣ Ejecución de la Simulación Principal

Este script realiza la derivación, calibración y simulación:

```bash
python modelo_cosmologico.py
```

---

### 3️⃣ Verificación del Blindaje Teórico

Ejecute este script para verificar simbólicamente la consistencia de la escala νₐ con la simetría del Grupo de Renormalización (Punto Fijo RG):

```bash
python analysis/derivacion_vc.py
```

Una vez completada la ejecución, los archivos de salida (`.tex`, `.png`, `.pdf`) se generarán automáticamente dentro de la carpeta `output/`.

---

## ⚙️ Ejecución del *Pipeline* Completo vía CLI

Absolutamente. La línea que señalas **es el comando CLI (Command Line Interface)**.
Esa es la instrucción exacta que usas en la terminal de Linux para iniciar la ejecución del *pipeline* del DVT, cargando todos los módulos y comenzando el muestreo **MCMC**.

---

### 🔍 Desglose del Comando CLI

El comando es complejo porque ejecuta un **paquete modular** (`genesis_modular`) y no un simple script.
Cada parámetro tiene un propósito específico:

| Fragmento del Comando                    | Propósito                                                                                               | Valor Usado                                                         |
| :--------------------------------------- | :------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------ |
| `python -m genesis_modular.run_pipeline` | **EJECUTABLE PRINCIPAL.** Ejecuta el módulo `run_pipeline` dentro del paquete `genesis_modular`.        | `run_pipeline`                                                      |
| `--gp "..."`                             | **Modelo GP (OBLIGATORIO).** Especifica el modelo del potencial escalar *(V(Φ))* previamente entrenado. | `"genesis_modular/dvt/dvt_ultra_results/model_20250909_165051.pkl"` |
| `--walkers 150`                          | **Parámetro MCMC.** Número de cadenas simultáneas para el muestreo.                                     | `150`                                                               |
| `--steps 15000`                          | **Parámetro MCMC.** Iteraciones por cada cadena.                                                        | `15000`                                                             |
| `--pool 6`                               | **Paralelización.** Núcleos de CPU utilizados para el cálculo del *likelihood*.                         | `6`                                                                 |
| `--thin 10`                              | **Post-procesamiento.** Adelgazamiento (solo guarda 1 de cada 10 pasos).                                | `10`                                                                |
| `--outdir "..."`                         | **Salida.** Carpeta donde se guardarán los resultados (cadenas, gráficos, resúmenes).                   | `"results/dvt_run_paper_con_gp"`                                    |

---

### 🚀 Ejemplo de Comando Completo

```bash
python -m genesis_modular.run_pipeline \
  --gp "genesis_modular/dvt/dvt_ultra_results/model_20250909_165051.pkl" \
  --walkers 150 \
  --steps 15000 \
  --pool 6 \
  --thin 10 \
  --outdir "results/dvt_run_paper_con_gp"
```

---

## 🧩 En Resumen

✅ Este repositorio implementa un **marco teórico falsable** para la Energía del Vacío Dinámica.
🔬 Incluye desde la derivación simbólica del modelo hasta la inferencia bayesiana completa.
📈 Todo el flujo —derivación, calibración, simulación y análisis MCMC— puede reproducirse desde terminal o notebooks.

---

**Autor:** Juan Galaz
**Proyecto:** Dynamic Vacuum Toolkit (DVT) – Geometría Causal-Informacional (GCI)
**Licencia:** MIT
