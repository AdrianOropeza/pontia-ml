from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


NOMBRES_MODELOS = {
    "regresion_logistica": "Regresión logística",
    "arbol_decision": "Árbol de decisión",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "red_neuronal": "Red neuronal (Keras)",
}

COLUMNAS_METRICAS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

METRICAS_PRINCIPALES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "f1-score": "F1",
    "roc_auc": "ROC-AUC",
    "roc-auc": "ROC-AUC",
    "auc": "ROC-AUC",
}


class ModelEvaluator:
    """Evalúa y compara clasificadores binarios ya entrenados.

    El evaluador está pensado para consumir directamente el diccionario de
    modelos devuelto por ``ModelTrainer.entrenar_todos``. Todos los modelos del
    proyecto exponen ``predict_proba``; por ello se usa una única lógica de
    evaluación, incluida la red neuronal Keras adaptada a la interfaz de
    scikit-learn.

    Parameters
    ----------
    threshold : float, default=0.5
        Umbral aplicado a P(is_canceled=1) para convertir probabilidades en
        predicciones de clase.
    primary_metric : str, default="f1"
        Métrica utilizada para seleccionar automáticamente el mejor modelo.
        Valores admitidos: accuracy, precision, recall, f1 y roc_auc.
    output_dir : str or pathlib.Path, default="outputs/evaluation"
        Directorio donde se guardan tablas y visualizaciones.
    """

    def __init__(
        self,
        threshold=0.5,
        primary_metric="f1",
        output_dir="outputs/evaluation",
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold debe estar entre 0 y 1.")

        metrica_normalizada = str(primary_metric).strip().lower()
        if metrica_normalizada not in METRICAS_PRINCIPALES:
            opciones = ", ".join(sorted(METRICAS_PRINCIPALES))
            raise ValueError(
                f"Métrica principal no válida: {primary_metric}. "
                f"Opciones: {opciones}."
            )

        self.threshold = float(threshold)
        self.primary_metric = METRICAS_PRINCIPALES[metrica_normalizada]
        self.output_dir = Path(output_dir)
        self.confusion_dir = self.output_dir / "confusion_matrices"
        self._crear_directorios()

        self.resultados_ = None
        self.detalles_ = {}
        self.claves_modelos_ = {}

    def _crear_directorios(self):
        """Crea los directorios de salida necesarios si no existen."""
        self.confusion_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validar_objetivo(y_real):
        """Comprueba que el target contiene exactamente las clases 0 y 1."""
        y_array = np.asarray(y_real).ravel()
        clases = np.unique(y_array)
        if not np.array_equal(clases, np.array([0, 1])):
            raise ValueError(
                "La evaluación requiere un target binario con clases 0 y 1. "
                f"Clases recibidas: {clases.tolist()}"
            )
        return y_array

    @staticmethod
    def _nombre_legible(clave_modelo):
        """Devuelve un nombre legible para mostrar el modelo en tablas y gráficos."""
        return NOMBRES_MODELOS.get(
            clave_modelo,
            str(clave_modelo).replace("_", " ").strip().title(),
        )

    @staticmethod
    def _extraer_probabilidad_positiva(modelo, X):
        """Obtiene P(clase=1) a partir de un clasificador entrenado."""
        if not hasattr(modelo, "predict_proba"):
            raise TypeError(
                "El modelo debe implementar predict_proba para calcular "
                "ROC-AUC y la curva ROC."
            )

        probabilidades = np.asarray(modelo.predict_proba(X))

        if probabilidades.ndim == 1:
            proba_positiva = probabilidades
        elif probabilidades.ndim == 2 and probabilidades.shape[1] >= 2:
            indice_positivo = 1
            clases = getattr(modelo, "classes_", None)
            if clases is not None:
                clases = np.asarray(clases)
                posiciones = np.where(clases == 1)[0]
                if posiciones.size == 1:
                    indice_positivo = int(posiciones[0])
            proba_positiva = probabilidades[:, indice_positivo]
        else:
            raise ValueError(
                "predict_proba debe devolver un vector de probabilidades o "
                "una matriz con una columna por clase."
            )

        proba_positiva = np.asarray(proba_positiva, dtype=float).ravel()
        if not np.all(np.isfinite(proba_positiva)):
            raise ValueError("Las probabilidades contienen valores no finitos.")
        if np.any((proba_positiva < 0.0) | (proba_positiva > 1.0)):
            raise ValueError("Las probabilidades deben estar entre 0 y 1.")

        return proba_positiva

    def evaluar_modelo(self, clave_modelo, modelo, X_eval, y_eval):
        """Evalúa un único modelo y almacena sus predicciones detalladas."""
        y_real = self._validar_objetivo(y_eval)
        y_proba = self._extraer_probabilidad_positiva(modelo, X_eval)

        if len(y_real) != len(y_proba):
            raise ValueError(
                "El número de probabilidades no coincide con el número de "
                "observaciones reales."
            )

        y_pred = (y_proba >= self.threshold).astype(int)
        matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])

        nombre = self._nombre_legible(clave_modelo)
        metricas = {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_real, y_pred),
            "Precision": precision_score(
                y_real, y_pred, pos_label=1, zero_division=0
            ),
            "Recall": recall_score(y_real, y_pred, pos_label=1, zero_division=0),
            "F1": f1_score(y_real, y_pred, pos_label=1, zero_division=0),
            "ROC-AUC": roc_auc_score(y_real, y_proba),
        }

        self.detalles_[clave_modelo] = {
            "nombre": nombre,
            "y_real": y_real,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "matriz_confusion": matriz,
            "metricas": metricas.copy(),
        }
        self.claves_modelos_[nombre] = clave_modelo
        return metricas

    def evaluar_todos(self, modelos, X_eval, y_eval, guardar_csv=True):
        """Evalúa todos los modelos con las mismas métricas y el mismo umbral.

        Parameters
        ----------
        modelos : dict
            Diccionario ``{nombre: modelo_entrenado}``. Es compatible de forma
            directa con la salida de ``ModelTrainer.entrenar_todos``.
        X_eval, y_eval : array-like
            Conjunto que se desea evaluar. Durante el desarrollo se recomienda
            usar validación y reservar test para la evaluación final.
        guardar_csv : bool, default=True
            Si es True, guarda la tabla comparativa en CSV.

        Returns
        -------
        pandas.DataFrame
            Tabla ordenada por la métrica principal y los criterios de desempate.
        """
        if not isinstance(modelos, dict) or not modelos:
            raise ValueError("modelos debe ser un diccionario no vacío.")

        self.detalles_ = {}
        self.claves_modelos_ = {}

        filas = []
        for clave_modelo, modelo in modelos.items():
            filas.append(
                self.evaluar_modelo(clave_modelo, modelo, X_eval, y_eval)
            )

        tabla = pd.DataFrame(filas).set_index("Modelo")
        criterios = self._criterios_ordenacion()
        tabla = tabla.sort_values(
            by=criterios,
            ascending=[False] * len(criterios),
            kind="stable",
        )
        self.resultados_ = tabla

        if guardar_csv:
            ruta_csv = self.output_dir / "model_comparison.csv"
            tabla.to_csv(ruta_csv, float_format="%.6f")

        return tabla.copy()

    def _criterios_ordenacion(self):
        """Devuelve la métrica principal y criterios transparentes de desempate."""
        criterios = [self.primary_metric]
        for metrica in ["ROC-AUC", "Recall"]:
            if metrica not in criterios:
                criterios.append(metrica)
        return criterios

    def seleccionar_mejor_modelo(self):
        """Selecciona el mejor resultado usando la métrica principal configurada.

        En caso de empate exacto se usa ROC-AUC y después Recall. La función
        devuelve la clave original del diccionario de ``ModelTrainer`` para que
        pueda recuperarse el objeto entrenado sin depender del nombre visual.
        """
        if self.resultados_ is None or self.resultados_.empty:
            raise RuntimeError(
                "Primero debe ejecutarse evaluar_todos antes de seleccionar."
            )

        criterios = self._criterios_ordenacion()
        tabla_ordenada = self.resultados_.sort_values(
            by=criterios,
            ascending=[False] * len(criterios),
            kind="stable",
        )
        nombre = tabla_ordenada.index[0]
        clave = self.claves_modelos_[nombre]
        fila = tabla_ordenada.iloc[0]

        return {
            "clave": clave,
            "nombre": nombre,
            "metrica_principal": self.primary_metric,
            "valor": float(fila[self.primary_metric]),
            "roc_auc": float(fila["ROC-AUC"]),
            "recall": float(fila["Recall"]),
        }

    def graficar_matrices_confusion(self):
        """Genera una matriz absoluta y otra normalizada para cada modelo."""
        self._comprobar_evaluacion()
        rutas = {}

        for clave_modelo, detalle in self.detalles_.items():
            y_real = detalle["y_real"]
            y_pred = detalle["y_pred"]
            nombre = detalle["nombre"]

            matriz_abs = confusion_matrix(y_real, y_pred, labels=[0, 1])
            matriz_norm = confusion_matrix(
                y_real, y_pred, labels=[0, 1], normalize="true"
            )

            figura, ejes = plt.subplots(1, 2, figsize=(11, 4.5))

            ConfusionMatrixDisplay(
                confusion_matrix=matriz_abs,
                display_labels=["No cancelada", "Cancelada"],
            ).plot(ax=ejes[0], colorbar=False, values_format="d")
            ejes[0].set_title("Valores absolutos")
            ejes[0].set_xlabel("Predicción")
            ejes[0].set_ylabel("Valor real")

            ConfusionMatrixDisplay(
                confusion_matrix=matriz_norm,
                display_labels=["No cancelada", "Cancelada"],
            ).plot(ax=ejes[1], colorbar=False, values_format=".2f")
            ejes[1].set_title("Normalizada por clase real")
            ejes[1].set_xlabel("Predicción")
            ejes[1].set_ylabel("Valor real")

            figura.suptitle(f"Matriz de confusión — {nombre}")
            figura.tight_layout()

            ruta = self.confusion_dir / f"{clave_modelo}.png"
            figura.savefig(ruta, dpi=160, bbox_inches="tight")
            plt.close(figura)
            rutas[clave_modelo] = ruta

        return rutas

    def graficar_roc_comparativa(self):
        """Dibuja en una sola figura la curva ROC de todos los modelos."""
        self._comprobar_evaluacion()

        figura, eje = plt.subplots(figsize=(8, 6))
        for detalle in self.detalles_.values():
            fpr, tpr, _ = roc_curve(detalle["y_real"], detalle["y_proba"])
            auc = detalle["metricas"]["ROC-AUC"]
            eje.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{detalle['nombre']} (AUC={auc:.3f})",
            )

        eje.plot([0, 1], [0, 1], linestyle="--", label="Clasificador aleatorio")
        eje.set_xlabel("Tasa de falsos positivos (FPR)")
        eje.set_ylabel("Tasa de verdaderos positivos (TPR)")
        eje.set_title("Curvas ROC comparativas")
        eje.legend(loc="lower right")
        eje.grid(alpha=0.25)
        figura.tight_layout()

        ruta = self.output_dir / "roc_comparison.png"
        figura.savefig(ruta, dpi=160, bbox_inches="tight")
        plt.close(figura)
        return ruta

    def graficar_comparacion_metricas(self):
        """Crea un mapa de calor con las métricas de todos los modelos."""
        self._comprobar_evaluacion()

        figura, eje = plt.subplots(
            figsize=(9, max(4.5, 0.75 * len(self.resultados_) + 2))
        )
        sns.heatmap(
            self.resultados_[COLUMNAS_METRICAS],
            annot=True,
            fmt=".3f",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "Valor de la métrica"},
            ax=eje,
        )
        eje.set_title("Comparación global de métricas")
        eje.set_xlabel("Métrica")
        eje.set_ylabel("Modelo")
        figura.tight_layout()

        ruta = self.output_dir / "metrics_comparison.png"
        figura.savefig(ruta, dpi=160, bbox_inches="tight")
        plt.close(figura)
        return ruta

    def graficar_importancia_random_forest(self, modelos, top_n=20):
        """Representa las variables transformadas más importantes del Random Forest.

        La función utiliza un Pipeline con pasos llamados ``preprocesado`` y
        ``modelo``. Además de la imagen, guarda un CSV con los valores de
        importancia calculados.
        """
        self._comprobar_evaluacion()
        if top_n <= 0:
            raise ValueError("top_n debe ser un entero positivo.")

        clave = self._buscar_random_forest(modelos)
        pipeline = modelos[clave]

        if not hasattr(pipeline, "named_steps"):
            raise TypeError(
                "Random Forest debe ser un Pipeline con los pasos "
                "'preprocesado' y 'modelo'."
            )

        preprocesador = pipeline.named_steps.get("preprocesado")
        estimador = pipeline.named_steps.get("modelo")
        if preprocesador is None or estimador is None:
            raise ValueError(
                "No se encontraron los pasos 'preprocesado' y 'modelo' "
                "esperados en el Pipeline de Random Forest."
            )
        if not hasattr(estimador, "feature_importances_"):
            raise TypeError(
                "El estimador de Random Forest no expone feature_importances_."
            )
        if not hasattr(preprocesador, "get_feature_names_out"):
            raise TypeError(
                "El preprocesador no permite recuperar los nombres de variables."
            )

        nombres = np.asarray(preprocesador.get_feature_names_out(), dtype=str)
        nombres = np.array([self._limpiar_nombre_feature(x) for x in nombres])
        importancias = np.asarray(estimador.feature_importances_, dtype=float)

        if len(nombres) != len(importancias):
            raise ValueError(
                "El número de nombres transformados no coincide con el número "
                "de importancias del Random Forest."
            )

        tabla = pd.DataFrame(
            {"Variable": nombres, "Importancia": importancias}
        ).sort_values("Importancia", ascending=False)

        ruta_csv = self.output_dir / "random_forest_feature_importance.csv"
        tabla.to_csv(ruta_csv, index=False, float_format="%.8f")

        top = tabla.head(min(top_n, len(tabla))).sort_values(
            "Importancia", ascending=True
        )
        figura, eje = plt.subplots(figsize=(9, max(5, 0.35 * len(top) + 2)))
        eje.barh(top["Variable"], top["Importancia"])
        eje.set_xlabel("Importancia")
        eje.set_ylabel("Variable transformada")
        eje.set_title(
            f"Random Forest — {len(top)} variables con mayor importancia"
        )
        eje.grid(axis="x", alpha=0.25)
        figura.tight_layout()

        ruta_png = self.output_dir / "random_forest_feature_importance.png"
        figura.savefig(ruta_png, dpi=160, bbox_inches="tight")
        plt.close(figura)

        return {"png": ruta_png, "csv": ruta_csv}

    @staticmethod
    def _limpiar_nombre_feature(nombre):
        """Elimina el prefijo técnico del ColumnTransformer para el gráfico."""
        return nombre.split("__", 1)[1] if "__" in nombre else nombre

    @staticmethod
    def _buscar_random_forest(modelos):
        if "random_forest" in modelos:
            return "random_forest"

        for clave in modelos:
            if "random" in str(clave).lower() and "forest" in str(clave).lower():
                return clave
        raise KeyError("No se encontró un modelo Random Forest en el diccionario.")

    def generar_visualizaciones(self, modelos, top_n_importancias=20):
        """Genera todas las visualizaciones definidas para la evaluación."""
        rutas = {
            "matrices_confusion": self.graficar_matrices_confusion(),
            "roc_comparativa": self.graficar_roc_comparativa(),
            "comparacion_metricas": self.graficar_comparacion_metricas(),
            "importancia_random_forest": self.graficar_importancia_random_forest(
                modelos, top_n=top_n_importancias
            ),
        }
        return rutas

    def evaluar_y_visualizar(
        self,
        modelos,
        X_eval,
        y_eval,
        guardar_csv=True,
        top_n_importancias=20,
    ):
        """Ejecuta de una vez evaluación, selección y visualizaciones.

        Es el punto de integración más directo con ``ModelTrainer``. Devuelve
        la tabla comparativa, la información del modelo seleccionado y las rutas
        de todos los artefactos generados.
        """
        tabla = self.evaluar_todos(
            modelos, X_eval, y_eval, guardar_csv=guardar_csv
        )
        mejor = self.seleccionar_mejor_modelo()
        rutas = self.generar_visualizaciones(
            modelos, top_n_importancias=top_n_importancias
        )
        return tabla, mejor, rutas

    def _comprobar_evaluacion(self):
        if self.resultados_ is None or not self.detalles_:
            raise RuntimeError(
                "Primero debe ejecutarse evaluar_todos antes de generar "
                "resultados o visualizaciones."
            )
