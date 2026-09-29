### Predicción de cancelaciones de reservas hoteleras

1. Justificación del problema

El objetivo del proyecto es predecir si una reserva hotelera será cancelada (is_canceled = 1) o no (is_canceled = 0) a partir de la información disponible en el momento de la reserva. Se trata, por tanto, de un problema de clasificación binaria.

El interés de negocio es claro: las cancelaciones generan pérdidas de ingresos y dificultan la gestión de la ocupación. Un modelo capaz de estimar la probabilidad de cancelación permitiría al hotel priorizar confirmaciones sobre reservas de riesgo.

El conjunto de datos utilizado contiene 119.390 reservas y 32 variables originales, con información sobre el cliente, el comportamiento de reserva y el resultado final. La variable objetivo está razonablemente representada (ver EDA), lo que hace el dataset adecuado para entrenar modelos de clasificación sin necesidad de recurrir a técnicas agresivas de reequilibrado.

2. Análisis exploratorio de datos (EDA)

El EDA se realizó sobre los datos ya cargados y limpios, con el objetivo de entender la naturaleza de las variables y orientar las decisiones de preprocesado y de modelado. A continuación se resumen los hallazgos principales.

2.1 Distribución de la variable objetivo

El dataset presenta un desbalanceo moderado: aproximadamente el 63 % de las reservas no se cancelan (≈75.000) frente a un 37 % que sí (≈44.000).

Este desbalanceo, aunque no severo, tiene una implicación directa en la elección de la métrica de evaluación: la accuracy por sí sola puede resultar engañosa, ya que un modelo podría obtener buenos resultados acertando la clase mayoritaria (no cancela) y fallando en la minoritaria (cancela), que es precisamente la de mayor interés para el negocio. Por ello se recomienda priorizar métricas como F1-score y AUC-ROC. (La elección definitiva de la métrica principal corresponde al equipo — coordinar con la persona C.)

2.2 Tipo de depósito (deposit_type)

El hallazgo más llamativo del EDA: las reservas con depósito "Non Refund" (no reembolsable) se cancelan casi en su totalidad, un comportamiento contraintuitivo (cabría esperar que quien paga un depósito no reembolsable no cancelara). Las causas exactas no pueden determinarse solo con los datos, pero apuntan a fenómenos como reservas de riesgo, políticas de overbooking o reservas no destinadas a cumplirse.

Conclusión: deposit_type es una de las variables con mayor poder predictivo del dataset.

2.3 Antelación de la reserva (lead_time)

Las reservas que se cancelan se realizaron, de media, con más del doble de antelación que las que no se cancelan (mediana ≈110 días frente a ≈45 días). Es coherente: a mayor antelación, más margen para que cambien los planes del cliente. lead_time es, por tanto, otra variable predictiva relevante.

2.4 Peticiones especiales y plazas de parking

Un mayor número de peticiones especiales (total_of_special_requests) y de plazas de parking solicitadas (required_car_parking_spaces) se asocia con una menor probabilidad de cancelación. Interpretación razonable: el cliente que se molesta en solicitar extras muestra una intención más firme de acudir.

2.5 Correlaciones generales

La matriz de correlación de variables numéricas muestra que ninguna variable numérica por sí sola presenta una correlación fuerte con la cancelación. El poder predictivo está repartido entre muchas variables y, de forma importante, en las variables categóricas (como deposit_type). Esto justifica el uso de modelos capaces de combinar múltiples variables e interacciones (Random Forest, Gradient Boosting) frente a modelos lineales simples.

3. Diseño del preprocesado

Todo el preprocesado se implementó de forma modular en src/data_loader.py, separado en funciones con responsabilidades claras y orquestado por la función preparar_datos(). A continuación se documentan las decisiones tomadas y su justificación.

3.1 Tratamiento de valores nulos

Solo cuatro variables presentaban valores faltantes, y cada una se trató según su naturaleza — no todos los nulos son iguales:

Variable	Nulos	Decisión	Justificación
company	112.593 (94 %)	Convertir a binaria has_company (1 = hay empresa, 0 = no) y eliminar la original	Con un 94 % de nulos, rellenar equivaldría a inventar la columna. El nulo tiene significado ("reserva de particular"), por lo que se conserva esa señal en forma binaria.
agent	16.340 (14 %)	Convertir a binaria has_agent y eliminar la original	333 agentes distintos (alta cardinalidad); los IDs concretos no aportan señal aprendible fiable. Lo relevante es si la reserva vino o no de agencia.
children	4	Rellenar con 0	Cantidad ínfima; se asume que la ausencia de dato equivale a "sin niños", la opción más conservadora (no introduce valores inventados altos).
country	488 (0,4 %)	Rellenar con "Unknown"	Es un dato genuinamente desconocido; se crea una categoría explícita en lugar de imputar un país inventado.
3.2 Eliminación de fugas de datos (data leakage)

Se eliminaron las columnas reservation_status y reservation_status_date. reservation_status contiene directamente el resultado de la reserva (incluido el valor "Canceled"), información que solo existe después de que la reserva se resuelva. Incluirla proporcionaría al modelo la respuesta que debe predecir, inflando artificialmente las métricas y produciendo un modelo inútil en un escenario real de predicción. Su eliminación garantiza que el modelo aprenda únicamente de información disponible en el momento de la reserva.

3.3 Alta cardinalidad en country

La variable country presentaba 178 valores distintos. Aplicar One-Hot Encoding directamente habría generado 178 columnas, la mayoría casi vacías. Se optó por conservar los 10 países más frecuentes y agrupar el resto bajo la etiqueta "Other", reduciendo la variable a 11 categorías sin perder la señal de los mercados principales (Portugal, Reino Unido, Francia, España, Alemania, etc.).

3.4 Codificación y escalado (ColumnTransformer)

El preprocesado final se encapsula en un ColumnTransformer que aplica un tratamiento diferenciado según el tipo de variable:

Variables numéricas → StandardScaler (estandarización a media 0 y desviación 1).
Variables categóricas → OneHotEncoder con handle_unknown="ignore", que evita errores si en fase de predicción aparece una categoría no vista durante el entrenamiento.

El uso de ColumnTransformer permite aplicar transformaciones distintas a cada grupo de columnas de forma ordenada y reutilizable, y encaja directamente en un Pipeline de scikit-learn junto al modelo, evitando data leakage entre entrenamiento y test (el ajuste se realiza solo sobre el conjunto de entrenamiento).

Tras el preprocesado, las 29 variables de entrada se transforman en 91 columnas numéricas listas para el modelado.

3.5 Partición de datos

Los datos se dividen en entrenamiento (80 %) y test (20 %) mediante train_test_split, con stratify=y para mantener la proporción de cancelaciones en ambos conjuntos, y random_state=42 para garantizar la reproducibilidad. (El equipo utiliza la misma semilla en todo el pipeline.)

4. Limitaciones y mejoras futuras (parte de preprocesado)
Binarización de agent y company: se optó por binarizar estas variables por su alta cardinalidad y el riesgo de sobreajuste. Como limitación, se pierde la posibilidad de capturar patrones de cancelación específicos por agencia o empresa. Una mejora futura sería conservar el top-N agentes/empresas más frecuentes (como se hizo con country) o aplicar target encoding para representar cada categoría por su tasa histórica de cancelación, capturando esos patrones sin explotar la dimensionalidad.
Variables temporales: arrival_date_month se trató como categórica con One-Hot, ignorando su orden natural y su carácter cíclico. Una mejora sería una codificación cíclica (seno/coseno) o un tratamiento ordinal que respete la secuencia de los meses.
Variables de fecha numéricas (arrival_date_year, arrival_date_week_number, arrival_date_day_of_month) se escalaron como magnitudes numéricas; un tratamiento más fino las consideraría categorías temporales.


5. Modelado y comparación de modelos (parte B)

5.1 Modelos entrenados

Se entrenaron cinco modelos de clasificación para predecir is_canceled: regresión logística, árbol de decisión, Random Forest, XGBoost y una red neuronal con Keras. Los cuatro primeros combinan el preprocesador de la parte A y el clasificador en un Pipeline. La red neuronal utiliza el mismo preprocesador, convierte su salida a datos numéricos densos y aplica un escalado adicional antes de entrenar. La clase ModelTrainer de src/model_trainer.py permite construir y entrenar los cinco modelos con una interfaz común.

5.2 Entrenamiento y búsqueda de hiperparámetros

Dentro del conjunto de entrenamiento se separó un 20 % para validación, manteniendo la proporción de cancelaciones. Primero se entrenó cada modelo con parámetros iniciales razonables. Después se aplicó GridSearchCV a los cinco modelos para comparar varias combinaciones de hiperparámetros. La búsqueda utilizó validación cruzada estratificada de 3 particiones y F1 como criterio de elección. Cada combinación se entrenó solo con los datos de ajuste; el conjunto de validación se utilizó después para comparar las predicciones. El conjunto de test quedó reservado para la evaluación final.

En la red neuronal se probaron distintas combinaciones de épocas y tamaño de lote. Su salida sigmoide estima la probabilidad de cancelación, que se convierte en clase usando un umbral de 0,5. Los experimentos y las tablas completas están en notebooks/modelos_exploracion.ipynb.

5.3 Resultados en validación

| Modelo | Parámetros elegidos por GridSearchCV | F1 inicial | F1 con GridSearchCV | ROC-AUC con GridSearchCV |
| --- | --- | ---: | ---: | ---: |
| Regresión logística | C=1 | 0,7263 | 0,7263 | 0,8948 |
| Árbol de decisión | max_depth=12; min_samples_leaf=10 | 0,7532 | 0,7827 | 0,9206 |
| Random Forest | max_depth=12; min_samples_leaf=5 | 0,7590 | 0,7617 | 0,9301 |
| XGBoost | max_depth=5; learning_rate=0,1 | 0,8077 | 0,8077 | 0,9408 |
| Red neuronal Keras | epochs=20; batch_size=128 | 0,8083 | 0,8092 | 0,9365 |

La mayor mejora de F1 se obtuvo con el árbol de decisión. Random Forest y la red neuronal mejoraron ligeramente; en regresión logística y XGBoost la búsqueda eligió los mismos valores que ya se habían usado inicialmente. Estos resultados corresponden a validación y sirven para comparar el modelado; la evaluación final con test y la selección del modelo definitivo corresponden a la parte C.