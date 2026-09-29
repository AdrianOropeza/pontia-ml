import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import preparar_features_inferencia


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIRECTORIO_MODELO_DEFAULT = RAIZ_PROYECTO / "models" / "best_model"
SALIDA_PREDICCIONES_DEFAULT = (
    RAIZ_PROYECTO / "outputs" / "predictions" / "predictions.csv"
)


class Predictor:
    """Carga el modelo final y realiza inferencia sobre reservas nuevas."""

    def __init__(self, model_dir=DIRECTORIO_MODELO_DEFAULT):
        self.model_dir = Path(model_dir)
        self.metadata = self._cargar_metadata()
        self.threshold = float(self.metadata["threshold"])
        self.model_type = self.metadata["model_type"]
        self.modelo = None
        self.preprocesador = None
        self.escalador = None
        self.batch_size = int(self.metadata.get("batch_size", 128))
        self._cargar_artefactos()

    def _cargar_metadata(self):
        ruta = self.model_dir / "metadata.json"
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró la metadata del modelo final en: {ruta}"
            )
        with ruta.open("r", encoding="utf-8") as archivo:
            metadata = json.load(archivo)

        claves = {
            "model_type",
            "threshold",
            "top_countries",
            "input_columns",
            "model_feature_columns",
            "artifacts",
        }
        faltantes = sorted(claves.difference(metadata))
        if faltantes:
            raise ValueError(
                "La metadata del modelo está incompleta. Faltan: "
                + ", ".join(faltantes)
            )
        return metadata

    def _cargar_artefactos(self):
        artefactos = self.metadata["artifacts"]

        if self.model_type == "keras":
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise ImportError(
                    "TensorFlow es necesario para cargar el modelo Keras final."
                ) from exc

            ruta_modelo = self.model_dir / artefactos["model"]
            ruta_preprocesado = self.model_dir / artefactos["preprocessing"]
            self.modelo = tf.keras.models.load_model(ruta_modelo)

            componentes = joblib.load(ruta_preprocesado)
            self.preprocesador = componentes["preprocesador"]
            self.escalador = componentes["escalador"]
            return

        if self.model_type == "sklearn_pipeline":
            ruta_modelo = self.model_dir / artefactos["model"]
            self.modelo = joblib.load(ruta_modelo)
            return

        raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

    def _preparar_datos(self, datos):
        return preparar_features_inferencia(
            datos,
            top_countries=self.metadata["top_countries"],
            columnas_entrada=self.metadata["input_columns"],
            columnas_modelo=self.metadata["model_feature_columns"],
        )

    @staticmethod
    def _probabilidad_positiva(probabilidades, clases=None):
        probabilidades = np.asarray(probabilidades)
        if probabilidades.ndim == 1:
            return probabilidades.astype(float)
        if probabilidades.ndim != 2 or probabilidades.shape[1] < 2:
            raise ValueError("predict_proba devolvió un formato no esperado.")

        indice = 1
        if clases is not None:
            clases = np.asarray(clases)
            posiciones = np.where(clases == 1)[0]
            if posiciones.size == 1:
                indice = int(posiciones[0])
        return probabilidades[:, indice].astype(float)

    def predict_proba(self, datos):
        """Devuelve P(is_canceled=1) para cada reserva recibida."""
        X = self._preparar_datos(datos)

        if self.model_type == "keras":
            X_transformado = self.preprocesador.transform(X)
            if hasattr(X_transformado, "toarray"):
                X_transformado = X_transformado.toarray()
            X_transformado = np.asarray(X_transformado, dtype=np.float32)
            X_transformado = self.escalador.transform(X_transformado).astype(
                np.float32
            )
            probabilidades = self.modelo.predict(
                X_transformado,
                batch_size=self.batch_size,
                verbose=0,
            ).ravel()
            return probabilidades.astype(float)

        if not hasattr(self.modelo, "predict_proba"):
            raise TypeError("El modelo guardado no implementa predict_proba.")
        probabilidades = self.modelo.predict_proba(X)
        clases = getattr(self.modelo, "classes_", None)
        return self._probabilidad_positiva(probabilidades, clases=clases)

    def predict(self, datos):
        """Devuelve clase y probabilidad de cancelación para cada reserva."""
        probabilidades = self.predict_proba(datos)
        predicciones = (probabilidades >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "prediction": predicciones,
                "cancellation_probability": probabilidades,
            },
            index=datos.index,
        )

    def predecir_dataframe(self, datos):
        """Añade las columnas de inferencia al DataFrame original."""
        resultados = self.predict(datos)
        salida = datos.copy()
        salida["prediction"] = resultados["prediction"]
        salida["cancellation_probability"] = resultados[
            "cancellation_probability"
        ]
        return salida


def _crear_parser():
    parser = argparse.ArgumentParser(
        description="Predice la probabilidad de cancelación de nuevas reservas."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del CSV con las reservas que se desean predecir.",
    )
    parser.add_argument(
        "--output",
        default=str(SALIDA_PREDICCIONES_DEFAULT),
        help="Ruta del CSV de salida con predicción y probabilidad.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DIRECTORIO_MODELO_DEFAULT),
        help="Directorio que contiene el modelo final y su metadata.",
    )
    return parser


def main():
    """Punto de entrada para realizar inferencia desde un CSV."""
    argumentos = _crear_parser().parse_args()
    ruta_entrada = Path(argumentos.input)
    ruta_salida = Path(argumentos.output)

    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No se encontró el CSV de entrada: {ruta_entrada}")

    datos = pd.read_csv(ruta_entrada)
    predictor = Predictor(argumentos.model_dir)
    resultados = predictor.predecir_dataframe(datos)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(ruta_salida, index=False)

    print(f"Modelo cargado: {predictor.metadata.get('model_name', 'modelo final')}")
    print(f"Reservas procesadas: {len(resultados)}")
    print(f"Threshold aplicado: {predictor.threshold}")
    print(f"Predicciones guardadas en: {ruta_salida}")


if __name__ == "__main__":
    main()
