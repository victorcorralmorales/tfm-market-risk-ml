# Estimación anticipada del riesgo de mercado mediante Machine Learning: comparativa de modelos supervisados sobre el S&P 500

Proyecto de **TFE** (Máster en Inteligencia Artificial). El objetivo no es predecir precios ni retornos, sino **anticipar episodios futuros de riesgo** en el mercado estadounidense usando el S&P 500 como referencia.

Modalidad: **comparativa de soluciones** (baselines, regresión logística, árboles) sobre un mismo target bien definido, con validación temporal e interpretación de errores.

---

## Objetivo

Construir un pipeline reproducible que:

1. Descargue y unifique datos de mercado (S&P 500, VIX, tipos y spreads).
2. Defina un target binario de riesgo a 20 días hábiles.
3. Particione el histórico en train / validation / test sin fuga de información.
4. Compare modelos supervisados frente a reglas simples (VIX).
5. Analice errores, importancia de variables y sensibilidad del target.

---

## Target principal

**`target_risk_20d` = 1** si en los **siguientes 20 días hábiles** se cumple al menos una condición:

- Drawdown futuro ≤ **−5%** (mínimo de cierres futuros respecto al cierre de hoy).
- Volatilidad futura anualizada ≥ **20%** (std de retornos diarios futuros × √252).

El target se calcula en `notebooks/02_build_dataset_with_target.ipynb`. Los notebooks 06 y 07 añaden análisis de interpretación y sensibilidad **sin sustituir** esta definición.

---

## Fuentes de datos

| Fuente | Variables |
|--------|-----------|
| [Yahoo Finance](https://finance.yahoo.com/) (`yfinance`) | OHLCV S&P 500 |
| [FRED](https://fred.stlouisfed.org/) (`pandas-datareader`) | VIX, tipos Treasury, spreads |

Datos crudos en `data/raw/`; datasets procesados en `data/processed/`.

---

## Estructura del repositorio

```
tfm-market-risk-ml/
├── notebooks/          # Pipeline completo (01 → 07)
├── data/
│   ├── raw/            # Descargas Yahoo / FRED
│   └── processed/      # Datasets y particiones de modelado
├── reports/
│   ├── figures/        # Gráficos (confusiones, ROC/PR, SHAP, sensibilidad)
│   └── tables/         # Métricas y tablas de resultados
├── docs/               # Documentación auxiliar (definición del target)
├── requirements.txt
└── README.md
```

---

## Pipeline de notebooks

| Notebook | Contenido |
|----------|-----------|
| `01_build_initial_dataset.ipynb` | Descarga y features de mercado |
| `02_target_definition_exploration.ipynb` | Exploración de definiciones de target |
| `02_build_dataset_with_target.ipynb` | Target definitivo `target_risk_20d` |
| `03_prepare_modeling_dataset.ipynb` | Anti-leakage, features, splits temporales |
| `04_baseline_models.ipynb` | Dummies, regla VIX, regresión logística |
| `05_tree_models.ipynb` | Random Forest y XGBoost |
| `06_error_analysis_interpretability.ipynb` | Errores, temporalidad, SHAP |
| `07_target_sensitivity_robustness.ipynb` | Sensibilidad de umbrales y horizonte |

**Partición temporal (principal):**

- Train: hasta 2015-12-31  
- Validation: 2016-02-02 → 2019-12-31 (con gap de purga de 20 días)  
- Test: desde 2020-01-31  

La selección de hiperparámetros y thresholds se hace **solo en validation**; test es evaluación final.

---

## Modelos evaluados

- Dummy (most frequent / stratified)
- Regla **VIX** (umbral calibrado en validation)
- **Regresión logística** (`all_features` y `reduced_no_level_features`)
- **Random Forest**
- **XGBoost**

## Métricas

Precision, recall, F1, ROC-AUC, PR-AUC, matrices de confusión, tasas de falsos positivos/negativos y análisis por año en test.

---

## Principales conclusiones (resumen)

- Hay **señal útil** para anticipar riesgo, pero la **generalización temporal es limitada**: el mejor modelo en validation no tiene por qué liderar en test.
- La regla **VIX** es un baseline financiero **muy fuerte** (también como score continuo).
- **XGBoost** obtiene el mejor F1 en validation; en test compiten VIX, LogReg reduced y Random Forest.
- Modelos **simples** (VIX, LogReg) siguen siendo competitivos; más complejidad no garantiza mejor test.
- Variables importantes coherentes: **VIX**, volatilidad realizada, drawdown.
- La **evaluación temporal** y la **interpretación de errores** (FN vs FP) son centrales para un uso prudente.

Detalle numérico en `reports/tables/model_comparison_summary.csv` y notebooks 04–07.

---

## Reproducir el entorno

Requisitos: Python 3.9+ (probado con 3.9).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                   # o jupyter notebook
```

Ejecutar los notebooks **en orden** (01 → 07). Los CSV en `data/processed/` y `reports/` permiten revisar resultados sin re-ejecutar todo.

### macOS y XGBoost

Si `import xgboost` falla por `libomp.dylib`:

```bash
brew install libomp
```

Alternativa sin Homebrew: runtime OpenMP de [mac.r-project.org/openmp](https://mac.r-project.org/openmp/) y configurar `DYLD_LIBRARY_PATH` o el rpath de `libxgboost.dylib`.

---

## Nota académica

Este repositorio corresponde a un **Trabajo Fin de Máster** en el marco del Máster en Inteligencia Artificial. El código y los resultados se ofrecen con fines académicos y de reproducibilidad; no constituyen asesoramiento financiero.

---

## Licencia y uso

Proyecto académico. Consultar con el autor antes de reutilizar en otros contextos.
