import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def cargar_datos(ruta):
    """Carga el dataset desde un CSV y lo devuelve como DataFrame."""
    df = pd.read_csv(ruta)
    return df

def limpiar_datos(df):
    """Limpia el dataset: trata nulos, transforma columnas y elimina fugas de datos (leakage)."""
    # company: 94% nulos -> binaria "hay empresa o no"
    df["has_company"] = df["company"].notnull().astype(int)
    df = df.drop(columns="company")

    # agent: 333 IDs distintos -> binaria "vino de agencia o no"
    df["has_agent"] = df["agent"].notnull().astype(int)
    df = df.drop(columns="agent")

    # children: solo 4 nulos, relleno con 0
    df["children"] = df["children"].fillna(0)

    # country: relleno nulos y agrupo en top-10 + "Other" (alta cardinalidad: 178 países)
    df["country"] = df["country"].fillna("Unknown")
    top10 = df["country"].value_counts().head(10).index
    df["country"] = df["country"].where(df["country"].isin(top10), "Other")

    # leakage: estas columnas contienen la respuesta -> se eliminan
    df = df.drop(columns=["reservation_status", "reservation_status_date"])

    return df

def separar_features_objetivo(df, objetivo="is_canceled"):
    """Separa el DataFrame en features (X) y variable objetivo (y)."""
    X = df.drop(columns=objetivo)
    y = df[objetivo]
    return X, y


def dividir_train_test(X, y, test_size=0.2, random_state=42):
    """Divide X e y en conjuntos de entrenamiento y prueba (estratificado)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


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

def preparar_datos(ruta):
    """Flujo completo: carga, limpia, separa, divide y construye el preprocesador.
    Devuelve los conjuntos train/test y el preprocesador listo para usar."""
    df = cargar_datos(ruta)
    df = limpiar_datos(df)
    X, y = separar_features_objetivo(df)
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)
    preprocesador = construir_preprocesador(X_train)
    return X_train, X_test, y_train, y_test, preprocesador