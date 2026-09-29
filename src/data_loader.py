import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


COLUMNAS_FUGA = ["reservation_status", "reservation_status_date"]


def cargar_datos(ruta):
    """Carga el dataset desde un CSV y lo devuelve como DataFrame."""
    df = pd.read_csv(ruta)
    return df


def limpiar_datos(df):
    """Limpia los datos y elimina variables que provocarían leakage.

    La función se reutiliza tanto durante entrenamiento como durante inferencia.
    Las columnas de resultado final se eliminan solo si están presentes, ya que
    una reserva nueva no tiene por qué incluirlas.
    """
    df = df.copy()

    # company: 94% nulos -> binaria "hay empresa o no"
    df["has_company"] = df["company"].notnull().astype(int)
    df = df.drop(columns="company")

    # agent: 333 IDs distintos -> binaria "vino de agencia o no"
    df["has_agent"] = df["agent"].notnull().astype(int)
    df = df.drop(columns="agent")

    # children: solo 4 nulos, relleno con 0
    df["children"] = df["children"].fillna(0)

    # country: relleno nulos; el top-10 se aprende después usando solo train
    df["country"] = df["country"].fillna("Unknown")

    # leakage: estas columnas contienen información posterior al resultado
    df = df.drop(columns=COLUMNAS_FUGA, errors="ignore")

    return df


def separar_features_objetivo(df, objetivo="is_canceled"):
    """Separa el DataFrame en features (X) y variable objetivo (y)."""
    if objetivo not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo '{objetivo}'.")
    X = df.drop(columns=objetivo)
    y = df[objetivo]
    return X, y


def dividir_train_test(X, y, test_size=0.2, random_state=42):
    """Divide X e y en conjuntos de entrenamiento y prueba (estratificado)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def obtener_top_countries(X, n=10):
    """Aprende los países más frecuentes a partir del conjunto recibido."""
    if "country" not in X.columns:
        raise ValueError("No se encontró la columna 'country'.")
    if n <= 0:
        raise ValueError("n debe ser un entero positivo.")
    return X["country"].value_counts().head(n).index.tolist()


def agrupar_country(X, top_countries):
    """Aplica una lista de países aprendida previamente y agrupa el resto."""
    if "country" not in X.columns:
        raise ValueError("No se encontró la columna 'country'.")
    if not top_countries:
        raise ValueError("top_countries no puede estar vacío.")

    X = X.copy()
    X["country"] = X["country"].where(
        X["country"].isin(top_countries), "Other"
    )
    return X


def construir_preprocesador(X):
    """Construye el ColumnTransformer: escala numéricas y codifica categóricas."""
    columnas_numericas = X.select_dtypes(include="number").columns.tolist()
    columnas_categoricas = X.select_dtypes(include="object").columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
        ]
    )
    return preprocesador


def preparar_features_inferencia(
    df,
    top_countries,
    columnas_entrada,
    columnas_modelo,
    objetivo="is_canceled",
):
    """Prepara reservas nuevas aplicando exactamente las reglas de entrenamiento.

    ``columnas_entrada`` y ``top_countries`` proceden de la metadata guardada al
    entrenar el modelo. El target y las columnas de resultado final se ignoran si
    aparecen en los datos recibidos, pero nunca se utilizan para predecir.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Los datos de inferencia deben proporcionarse como DataFrame.")

    datos = df.copy()
    datos = datos.drop(columns=[objetivo, *COLUMNAS_FUGA], errors="ignore")

    faltantes = [col for col in columnas_entrada if col not in datos.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias para realizar la predicción: "
            + ", ".join(faltantes)
        )

    # Se ignoran columnas adicionales y se conserva el orden esperado.
    datos = datos.loc[:, columnas_entrada]
    datos = limpiar_datos(datos)
    datos = agrupar_country(datos, top_countries)

    faltantes_modelo = [col for col in columnas_modelo if col not in datos.columns]
    if faltantes_modelo:
        raise ValueError(
            "Tras el preprocesamiento faltan columnas esperadas por el modelo: "
            + ", ".join(faltantes_modelo)
        )

    return datos.loc[:, columnas_modelo]


def preparar_datos(ruta, devolver_metadata=False):
    """Carga, limpia, divide y prepara el preprocesamiento del proyecto.

    Devuelve los conjuntos de entrenamiento y test junto con el preprocesador.
    Si ``devolver_metadata=True``, añade la información necesaria para reproducir
    el mismo tratamiento de datos durante la inferencia.
    """
    df_original = cargar_datos(ruta)
    columnas_entrada = [
        col
        for col in df_original.columns
        if col not in {"is_canceled", *COLUMNAS_FUGA}
    ]

    df = limpiar_datos(df_original)
    X, y = separar_features_objetivo(df)
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)

    # Agrupar country en top-10 + "Other" usando solo la distribución de train
    top10_countries = obtener_top_countries(X_train, n=10)
    X_train = agrupar_country(X_train, top10_countries)
    X_test = agrupar_country(X_test, top10_countries)

    preprocesador = construir_preprocesador(X_train)
    salida = (X_train, X_test, y_train, y_test, preprocesador)

    if not devolver_metadata:
        return salida

    metadata = {
        "top_countries": [str(pais) for pais in top10_countries],
        "input_columns": columnas_entrada,
        "model_feature_columns": X_train.columns.tolist(),
    }
    return (*salida, metadata)
