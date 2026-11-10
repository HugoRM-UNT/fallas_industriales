# Guía de sustentación (cada integrante: 1 decisión de preprocesamiento + 1 hiperparámetro + lógica de selección con CV)

## Decisiones de preprocesamiento (elija una)
- **Excluir TWF/HDF/PWF/OSF/RNF:** describen la causa de la falla; el objetivo es su O lógico. Usarlas sería fuga.
- **Excluir UDI y Product ID:** identificadores sin significado físico; permiten memorizar.
- **Imputación dentro del Pipeline:** estadísticos aprendidos solo con Train (y por fold en CV) -> sin fuga; también protege en producción.
- **Sin escalado:** los árboles son invariantes a transformaciones monótonas.
- **OneHot con [L,M,H] y handle_unknown='ignore':** sin orden falso; tolera categorías nuevas.
- **Split estratificado:** conserva ~3-4% de fallas en Train y Test.

## Hiperparámetros (elija uno)
- **max_depth / min_samples_leaf / ccp_alpha:** limitan la complejidad del árbol; menos varianza, menos sobreajuste.
- **n_estimators (RF):** más árboles reducen varianza hasta saturar; cuesta tiempo.
- **max_features:** decorrelaciona árboles del bosque.
- **bootstrap:** muestreo con reemplazo por árbol.
- **learning_rate + n_estimators (GB):** compromiso; tasa baja requiere más árboles.
- **subsample (GB):** boosting estocástico; reduce varianza.

## Lógica de selección con CV
1. Métrica principal fijada antes (PR-AUC); recall como métrica operativa.
2. Mismos 5 folds para los 3 modelos; RandomizedSearchCV solo en Train (20 configs).
3. Mejor configuración por modelo = mayor PR-AUC media en CV; ganador = mejor entre los tres.
4. Refit en todo Train; umbral por costo en predicciones OOF; Test una sola vez.
5. Si se mirara Test para elegir, el resultado sería optimista (sesgo de selección).

## Preguntas probables
- ¿Por qué PR-AUC y no accuracy? Desbalance (~4% fallas).
- ¿Por qué el umbral no es 0.5? Se minimiza el costo FN x10 + FP x1 en validación.
- ¿Qué tan estable es la ventaja? Ver diferencias por fold (informe, sección 5 y 8).
- ¿Qué cambiarían en planta real? Ver informe, sección 8.
