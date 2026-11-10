# Grupo 09 - Detección de fallas en equipos industriales
Decision Tree vs Random Forest vs Gradient Boosting (AI4I 2020), con Pipeline, CV de 5 folds y RandomizedSearchCV.

## Estado de los datos
Los resultados incluidos se generaron con el **CSV real** `ai4i2020.csv` (10 000 filas) entregado por el grupo.
Pendiente: completar los integrantes y confirmar la fecha de descarga y la licencia de Kaggle en `dataset_info.json`, y luego `python run_all.py`
para regenerar informe y presentación. (`generate_synthetic_data.py` es opcional: sólo sirve para pruebas sin el CSV.)

## Uso
```
pip install -r requirements.txt
python run_all.py              # entrena (≈4-10 min), figuras, informe PDF y PPTX
streamlit run app.py           # app: página 0 entrena desde el navegador; 1-6 muestran resultados y predicen
```
`python train.py --n-iter 5` hace una prueba rápida. El entrenamiento guarda checkpoints en `model/cache/` (bórrelo al cambiar datos o código del modelo).

## Contenido
| Ruta | Qué es |
|---|---|
| train.py | Etapas 02-16: EDA, split, Pipelines, CV base, búsqueda, ganador, umbral, Test, interpretación, 5 registros, joblib |
| app.py | Streamlit (6 páginas) |
| make_report.py / make_slides.py | Generan informe PDF y presentación (15 diapositivas) desde `model/results.json` |
| informe/, presentacion/ | PDF y PPTX generados |
| model/ | pipeline.joblib, results.json, cv_results_*.csv, nuevos_registros.csv |
| figures/ | Gráficos |
| docs/GUIA_SUSTENTACION.md | Guía para que cada integrante explique su decisión |

## Decisiones clave
Semilla 42; split 80/20 estratificado; mismos 5 folds; sample_weight balanceado en los 3 modelos; métrica principal PR-AUC (fijada antes);
umbral por costo mínimo (FN:FP = 10:1, supuesto) en predicciones fuera de muestra de Train; Test evaluado una sola vez.
Se excluyen TWF/HDF/PWF/OSF/RNF (fuga), UDI y Product ID.

## Entrenar desde la app
En `streamlit run app.py` la página **0. Entrenamiento** permite elegir `n_iter`, subir un CSV propio, reentrenar desde cero y
regenerar informe/presentación; muestra el registro en vivo. El entrenamiento corre SOLO al pulsar el botón; el formulario
predictivo sigue cargando el Pipeline guardado (no busca hiperparámetros en cada consulta). Ojo: entrenar sobrescribe `model/`.
