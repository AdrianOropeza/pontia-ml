import pandas as pd

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