import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import preparar_datos
from src.evaluator import ModelEvaluator
from src.model_trainer import ClasificadorKeras, ModelTrainer


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATASET = RAIZ_PROYECTO / "data" / "raw" / "dataset_practica_final.csv"
SALIDA_VALIDACION = RAIZ_PROYECTO / "outputs" / "evaluation"
SALIDA_TEST = RAIZ_PROYECTO / "outputs" / "final_evaluation"
RUTA_RESUMEN_FINAL = SALIDA_TEST / "final_evaluation_summary.json"
DIRECTORIO_MODELO_FINAL = RAIZ_PROYECTO / "models" / "best_model"

RANDOM_STATE = 42
TEST_SIZE_VALIDACION = 0.2
THRESHOLD = 0.5
METRICA_PRINCIPAL = "f1"


def dividir_desarrollo(X_train, y_train):
    """Divide el train externo en ajuste y validación de forma estratificada."""
    return train_test_split(
        X_train,
        y_train,
        test_size=TEST_SIZE_VALIDACION,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )


def seleccionar_modelo_en_validacion(
    X_ajuste,
    X_val,
    y_ajuste,
    y_val,
    preprocesador,
):
    """Entrena en ajuste y selecciona el modelo usando solo validación."""
    entrenador = ModelTrainer(
        preprocesador=preprocesador,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    modelos = entrenador.entrenar_todos(
        X_ajuste,
        y_ajuste,
        X_val=X_val,
        y_val=y_val,
    )

    evaluador = ModelEvaluator(
        threshold=THRESHOLD,
        primary_metric=METRICA_PRINCIPAL,
        output_dir=SALIDA_VALIDACION,
    )
    tabla, mejor_modelo, rutas = evaluador.evaluar_y_visualizar(
        modelos,
        X_val,
        y_val,
    )
    return tabla, mejor_modelo, rutas


def entrenar_modelos_finales(X_train, y_train, preprocesador):
    """Reentrena desde cero los cinco modelos usando todo el train externo."""
    entrenador = ModelTrainer(
        preprocesador=preprocesador,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    return entrenador.entrenar_todos(X_train, y_train)


def evaluar_modelos_en_test(modelos, X_test, y_test):
    """Evalúa los modelos finales en test sin realizar una nueva selección."""
    evaluador = ModelEvaluator(
        threshold=THRESHOLD,
        primary_metric=METRICA_PRINCIPAL,
        output_dir=SALIDA_TEST,
    )
    tabla = evaluador.evaluar_todos(modelos, X_test, y_test)
    rutas = evaluador.generar_visualizaciones(modelos)
    return tabla, rutas


def _limpiar_artefactos_modelo(directorio):
    """Elimina únicamente artefactos conocidos de una ejecución anterior."""
    directorio.mkdir(parents=True, exist_ok=True)
    for nombre in [
        "model.keras",
        "model.joblib",
        "preprocessing.joblib",
        "metadata.json",
    ]:
        ruta = directorio / nombre
        if ruta.exists():
            ruta.unlink()


def guardar_modelo_final(
    modelo,
    clave_modelo,
    nombre_modelo,
    metadata_datos,
    directorio=DIRECTORIO_MODELO_FINAL,
):
    """Guarda el modelo elegido y la metadata necesaria para inferencia."""
    directorio = Path(directorio)
    _limpiar_artefactos_modelo(directorio)

    artefactos = {}
    if isinstance(modelo, ClasificadorKeras):
        ruta_modelo = directorio / "model.keras"
        ruta_preprocesado = directorio / "preprocessing.joblib"

        modelo.modelo_.save(ruta_modelo)
        joblib.dump(
            {
                "preprocesador": modelo.preprocesador_,
                "escalador": modelo.escalador_,
            },
            ruta_preprocesado,
        )
        tipo_modelo = "keras"
        artefactos = {
            "model": ruta_modelo.name,
            "preprocessing": ruta_preprocesado.name,
        }
        batch_size = int(modelo.batch_size)
    else:
        ruta_modelo = directorio / "model.joblib"
        joblib.dump(modelo, ruta_modelo)
        tipo_modelo = "sklearn_pipeline"
        artefactos = {"model": ruta_modelo.name}
        batch_size = None

    metadata = {
        "model_key": clave_modelo,
        "model_name": nombre_modelo,
        "model_type": tipo_modelo,
        "primary_metric": METRICA_PRINCIPAL,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "top_countries": metadata_datos["top_countries"],
        "input_columns": metadata_datos["input_columns"],
        "model_feature_columns": metadata_datos["model_feature_columns"],
        "artifacts": artefactos,
    }
    if batch_size is not None:
        metadata["batch_size"] = batch_size

    ruta_metadata = directorio / "metadata.json"
    with ruta_metadata.open("w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, ensure_ascii=False, indent=2)

    return ruta_metadata


def _metricas_a_dict(fila):
    """Convierte una fila de métricas de pandas en valores JSON nativos."""
    return {str(clave): float(valor) for clave, valor in fila.items()}


def guardar_resumen_evaluacion(
    tabla_val,
    tabla_test,
    mejor_modelo,
    tamanos,
    ruta=RUTA_RESUMEN_FINAL,
):
    """Guarda la trazabilidad de selección y evaluación final del experimento."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    nombre = mejor_modelo["nombre"]
    resumen = {
        "selection": {
            "model_key": mejor_modelo["clave"],
            "model_name": nombre,
            "primary_metric": mejor_modelo["metrica_principal"],
            "threshold": THRESHOLD,
            "validation_metrics": _metricas_a_dict(tabla_val.loc[nombre]),
            "selected_before_test": True,
        },
        "final_test": {
            "metrics": _metricas_a_dict(tabla_test.loc[nombre]),
            "used_for_model_selection": False,
        },
        "configuration": {
            "random_state": RANDOM_STATE,
            "validation_size": TEST_SIZE_VALIDACION,
        },
        "dataset_sizes": tamanos,
    }

    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)

    return ruta


def main():
    """Ejecuta selección, entrenamiento final, test y persistencia del ganador."""
    print("========================================")
    print("PREPARACIÓN DE DATOS")
    print("========================================")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocesador,
        metadata_datos,
    ) = preparar_datos(RUTA_DATASET, devolver_metadata=True)

    print(f"Train reservado para desarrollo: {X_train.shape}")
    print(f"Test FINAL reservado:             {X_test.shape}")

    X_ajuste, X_val, y_ajuste, y_val = dividir_desarrollo(X_train, y_train)
    print(f"Ajuste:                           {X_ajuste.shape}")
    print(f"Validación:                       {X_val.shape}")

    print("\n========================================")
    print("SELECCIÓN EN VALIDACIÓN")
    print("========================================")
    tabla_val, mejor_modelo, rutas_val = seleccionar_modelo_en_validacion(
        X_ajuste,
        X_val,
        y_ajuste,
        y_val,
        preprocesador,
    )
    print(tabla_val.round(4))
    print("\nModelo seleccionado usando validación:")
    print(mejor_modelo)
    print("\nArtefactos de validación:")
    print(rutas_val)

    clave_seleccionada = mejor_modelo["clave"]
    nombre_seleccionado = mejor_modelo["nombre"]

    print("\n========================================")
    print("ENTRENAMIENTO FINAL")
    print("========================================")
    print(
        "Los cinco modelos se reentrenan desde cero con todo X_train. "
        "El conjunto de test todavía no interviene en el entrenamiento."
    )
    modelos_finales = entrenar_modelos_finales(
        X_train,
        y_train,
        preprocesador,
    )

    ruta_metadata_modelo = guardar_modelo_final(
        modelos_finales[clave_seleccionada],
        clave_seleccionada,
        nombre_seleccionado,
        metadata_datos,
    )
    print(f"\nModelo final guardado en: {DIRECTORIO_MODELO_FINAL}")
    print(f"Metadata de inferencia:   {ruta_metadata_modelo}")

    print("\n========================================")
    print("EVALUACIÓN FINAL SOBRE TEST")
    print("========================================")
    tabla_test, rutas_test = evaluar_modelos_en_test(
        modelos_finales,
        X_test,
        y_test,
    )
    print(tabla_test.round(4))

    print("\nModelo fijado previamente en validación:")
    print(f" - {nombre_seleccionado} ({clave_seleccionada})")
    print("\nMétricas finales de ese modelo en test:")
    print(tabla_test.loc[nombre_seleccionado].round(4))

    tamanos = {
        "train": int(len(X_train)),
        "adjustment": int(len(X_ajuste)),
        "validation": int(len(X_val)),
        "test": int(len(X_test)),
    }
    ruta_resumen = guardar_resumen_evaluacion(
        tabla_val,
        tabla_test,
        mejor_modelo,
        tamanos,
    )

    print("\nArtefactos finales:")
    print(rutas_test)
    print(f"\nResumen final guardado en: {ruta_resumen}")
    print(f"Resultados guardados en:   {SALIDA_TEST}")
    print(
        "IMPORTANTE: los resultados de test se usan únicamente para la "
        "evaluación final; no se vuelve a seleccionar ni ajustar el modelo."
    )


if __name__ == "__main__":
    main()
