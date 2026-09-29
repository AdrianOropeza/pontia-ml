# Metodología de evaluación — Persona C

Este documento describe la capa de evaluación implementada en `src/evaluator.py`. En esta fase no se modifican todavía el preprocesamiento de Persona A ni los hiperparámetros de Persona B. El objetivo es dejar una evaluación reutilizable que funcione igual aunque posteriormente se repita el entrenamiento con una configuración corregida.

## 1. Integración con el trabajo de A y B

`ModelEvaluator` consume directamente el diccionario de modelos que devuelve `ModelTrainer.entrenar_todos(...)`. Los cuatro modelos clásicos se entregan como `Pipeline` de scikit-learn y la red neuronal usa `ClasificadorKeras`, que expone `predict_proba()` con el mismo formato. Por ello los cinco modelos se evalúan mediante una interfaz común.

Ejemplo de integración:

```python
from src.evaluator import ModelEvaluator

# `modelos` es la salida de ModelTrainer.entrenar_todos(...)
evaluador = ModelEvaluator(
    threshold=0.5,
    primary_metric="f1",
)

tabla, mejor, rutas = evaluador.evaluar_y_visualizar(
    modelos,
    X_val,
    y_val,
)

print(tabla.round(3))
print(mejor)
```

Durante el desarrollo se recomienda ejecutar este bloque sobre el conjunto de validación. El conjunto de test debe reservarse para la evaluación final una vez estén congeladas las decisiones de preprocesamiento e hiperparámetros.

## 2. Predicción y umbral

Todos los modelos producen la probabilidad de la clase positiva (`is_canceled = 1`). Para asegurar una comparación homogénea se aplica el mismo umbral a todos:

```text
P(cancelación) >= 0.5  ->  predicción = 1
P(cancelación) <  0.5  ->  predicción = 0
```

El umbral se puede configurar en `ModelEvaluator`, pero no debe modificarse utilizando el conjunto de test.

## 3. Métricas

La tabla comparativa contiene las siguientes métricas:

- **Accuracy**: proporción total de aciertos.
- **Precision**: de las reservas predichas como canceladas, qué proporción termina cancelándose.
- **Recall**: de las reservas que realmente se cancelan, qué proporción detecta el modelo.
- **F1-score**: media armónica de precision y recall.
- **ROC-AUC**: capacidad del modelo para discriminar entre ambas clases a lo largo de distintos umbrales.

La métrica principal configurada es **F1-score**. El dataset presenta un desbalance moderado y no se ha definido un coste de negocio que justifique priorizar exclusivamente precision o recall. F1 permite valorar de forma conjunta ambas métricas. ROC-AUC se utiliza como métrica secundaria especialmente relevante porque evalúa la discriminación usando probabilidades y no depende de un único umbral de clasificación.

En caso de empate exacto en F1, la selección automática utiliza primero ROC-AUC y después Recall como criterios de desempate. Esta regla queda explícita y reproducible en el código.

## 4. Interpretación de la matriz de confusión

La clase positiva es la cancelación:

```text
TP: se predice cancelación y la reserva realmente cancela.
TN: se predice no cancelación y la reserva realmente se mantiene.
FP: se predice cancelación, pero la reserva finalmente se mantiene.
FN: se predice que la reserva se mantendrá, pero finalmente cancela.
```

Para cada modelo se generan dos matrices en una misma figura: una con valores absolutos y otra normalizada por la clase real. Esto permite interpretar tanto el volumen de errores como su proporción.

## 5. Visualizaciones generadas

Tras ejecutar `evaluar_y_visualizar(...)` se crean automáticamente:

```text
outputs/evaluation/
├── model_comparison.csv
├── metrics_comparison.png
├── roc_comparison.png
├── random_forest_feature_importance.csv
├── random_forest_feature_importance.png
└── confusion_matrices/
    ├── regresion_logistica.png
    ├── arbol_decision.png
    ├── random_forest.png
    ├── xgboost.png
    └── red_neuronal.png
```

`metrics_comparison.png` resume las cinco métricas en un mapa de calor. `roc_comparison.png` presenta las cinco curvas ROC en una misma figura. La importancia de variables se obtiene del `feature_importances_` del Random Forest ya entrenado y se vincula con los nombres producidos por el `ColumnTransformer`.

La importancia de una variable indica su contribución al comportamiento del Random Forest; no debe interpretarse como causalidad.

## 6. Política para el conjunto de test

En esta fase el evaluador queda implementado y puede probarse sobre validación. Los resultados finales todavía no deben redactarse, porque siguen pendientes dos decisiones del pipeline: trasladar al código final los hiperparámetros seleccionados durante la exploración y decidir si se corrige la selección del top-10 de `country` antes del split.

Una vez congeladas esas decisiones se repetirá, si procede, el entrenamiento y se ejecutará la evaluación sobre `X_test`/`y_test`. Esos resultados serán los utilizados en el README, el informe final y las conclusiones.
