import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier


def convertir_a_denso_float32(X):
    """Convierte la salida del ColumnTransformer a la entrada de Keras."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


class ClasificadorKeras(ClassifierMixin, BaseEstimator):
    """Red Keras con la misma interfaz de predicción que sklearn."""

    def __init__(
        self,
        preprocesador,
        random_state=42,
        epochs=20,
        batch_size=128,
        patience=3,
        verbose=0,
    ):
        self.preprocesador = preprocesador
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        """Entrena la red; usa parada temprana si recibe validación."""
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val e y_val deben proporcionarse juntos.")
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError("La red requiere etiquetas binarias 0 y 1 en y.")

        # Ajustar una copia del preprocesador solo con entrenamiento
        self.preprocesador_ = clone(self.preprocesador)
        X_red = convertir_a_denso_float32(
            self.preprocesador_.fit_transform(X, y)
        )

        # Aplicar un escalado adicional antes del entrenamiento de la red
        self.escalador_ = StandardScaler()
        X_red = self.escalador_.fit_transform(X_red).astype(np.float32)

        validation_data = None
        callbacks = []
        if X_val is not None:
            X_val_red = self._transformar(X_val)
            validation_data = (X_val_red, np.asarray(y_val))
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            ]

        tf.keras.utils.set_random_seed(self.random_state)
        self.modelo_ = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_red.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        self.modelo_.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.historial_ = self.modelo_.fit(
            X_red,
            np.asarray(y),
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        return self

    def _transformar(self, X):
        """Aplica los transformadores aprendidos; no vuelve a ajustarlos."""
        X_red = convertir_a_denso_float32(self.preprocesador_.transform(X))
        return self.escalador_.transform(X_red).astype(np.float32)

    def predict_proba(self, X):
        """Devuelve probabilidades con el formato (n_filas, 2)."""
        check_is_fitted(self, ["modelo_", "historial_", "classes_"])
        probabilidades = self.modelo_.predict(
            self._transformar(X), batch_size=self.batch_size, verbose=0
        ).ravel()
        return np.column_stack((1.0 - probabilidades, probabilidades))

    def predict(self, X):
        """Convierte P(cancelación) en etiquetas con umbral fijo de 0,5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ModelTrainer:
    """Construye y entrena los cinco modelos con un preprocesador común."""

    def __init__(
        self,
        preprocesador,
        random_state=42,
        epochs=20,
        batch_size=128,
        patience=3,
        verbose=0,
    ):
        estimadores = {
            "regresion_logistica": LogisticRegression(max_iter=1000),
            "arbol_decision": DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=10, random_state=random_state
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=4,
            ),
        }
        self.modelos = {
            nombre: Pipeline([
                ("preprocesado", clone(preprocesador)),
                ("modelo", estimador),
            ])
            for nombre, estimador in estimadores.items()
        }
        self.modelos["red_neuronal"] = ClasificadorKeras(
            preprocesador=preprocesador,
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=verbose,
        )

    def entrenar_modelo(self, nombre, X_train, y_train, X_val=None, y_val=None):
        """Entrena un modelo por su nombre y devuelve el objeto entrenado."""
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val e y_val deben proporcionarse juntos.")
        if nombre not in self.modelos:
            raise ValueError(
                f"Modelo desconocido: {nombre}. Opciones: {list(self.modelos)}"
            )
        modelo = self.modelos[nombre]
        if nombre == "red_neuronal":
            modelo.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            modelo.fit(X_train, y_train)
        return modelo

    def entrenar_todos(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena y devuelve los cinco modelos configurados."""
        for nombre in self.modelos:
            self.entrenar_modelo(nombre, X_train, y_train, X_val, y_val)
        return self.modelos
