"""
core_dev1.py — Funciones de EDA de Developer 1.

Contiene las dos funciones asignadas a Developer 1:
    - describe_df
    - tipifica_variables

NOTA DE INTEGRACIÓN (Scrum Master): el contenido de este módulo está pensado
para fusionarse tal cual en `toolbox_ml/eda/core.py`. Los tests de
`tests/test_core_dev1.py` ya importan desde `toolbox_ml.eda.core`, de modo que
pasarán a verde en cuanto estas funciones se muevan a `core.py`.
"""

import pandas as pd


def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.

    Devuelve un DataFrame con una fila por cada columna del DataFrame de
    entrada. El índice del resultado es el nombre de cada columna.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        columnas:
            - 'tipo': tipo de dato de la columna (object, int64, float64, ...).
            - 'porcentaje_nulos': % de valores nulos/NaN sobre el total de filas.
            - 'valores_unicos': número de valores únicos distintos.
            - 'porcentaje_cardinalidad': % de valores únicos sobre el total de filas.
        Retorna None si el input no es un pd.DataFrame válido.
    """
    # --- Comprobación de entrada -------------------------------------------
    # Si no es un DataFrame, informamos y devolvemos None (no lanzamos excepción).
    if not isinstance(df, pd.DataFrame):
        print(
            "Error: el argumento 'df' debe ser un pandas.DataFrame, "
            f"se recibió {type(df).__name__}."
        )
        return None

    # Número total de filas. Lo usamos como denominador de los porcentajes.
    n_filas = len(df)

    # --- Cálculo por columna -----------------------------------------------
    # Construimos un diccionario {columna: {métrica: valor}} y luego lo
    # convertimos en DataFrame transpuesto para tener una fila por columna.
    resumen = {}
    for col in df.columns:
        serie = df[col]

        # Tipo de dato de la columna (como string legible: 'int64', 'object', ...).
        tipo = serie.dtype

        # nunique cuenta valores únicos ignorando NaN por defecto.
        valores_unicos = serie.nunique(dropna=True)

        # Si el DataFrame está vacío evitamos dividir por cero: los porcentajes
        # quedan a 0.0 por convención.
        if n_filas == 0:
            porcentaje_nulos = 0.0
            porcentaje_cardinalidad = 0.0
        else:
            # isna().mean() da la proporción de nulos; *100 lo pasa a porcentaje.
            porcentaje_nulos = serie.isna().mean() * 100
            porcentaje_cardinalidad = valores_unicos / n_filas * 100

        resumen[col] = {
            "tipo": tipo,
            "porcentaje_nulos": porcentaje_nulos,
            "valores_unicos": valores_unicos,
            "porcentaje_cardinalidad": porcentaje_cardinalidad,
        }

    # orient='index' hace que cada clave (columna original) sea una fila.
    resultado = pd.DataFrame.from_dict(resumen, orient="index")

    # Garantizamos el orden de columnas exigido en el enunciado.
    resultado = resultado[
        ["tipo", "porcentaje_nulos", "valores_unicos", "porcentaje_cardinalidad"]
    ]
    return resultado


def tipifica_variables(
    df: pd.DataFrame,
    umbral_categoria: int,
    umbral_continua: float,
) -> pd.DataFrame:
    """
    Sugiere un tipo para cada variable del DataFrame según su cardinalidad.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categoria (int): entero positivo. Cardinalidad por debajo de la
            cual una variable se considera categórica.
        umbral_continua (float): float entre 0 y 100. Porcentaje de cardinalidad
            a partir del cual una variable numérica se considera continua.

    Retorna:
        pd.DataFrame: DataFrame con dos columnas, 'nombre_variable' y
        'tipo_sugerido', con una fila por cada columna del DataFrame de entrada.
        El 'tipo_sugerido' sigue esta cascada:
            - cardinalidad == 2                                  -> "Binaria"
            - cardinalidad < umbral_categoria                    -> "Categórica"
            - card. >= umbral_categoria y %card. >= umbral_continua -> "Numérica Continua"
            - card. >= umbral_categoria y %card. <  umbral_continua -> "Numérica Discreta"
        Retorna None si alguna comprobación de entrada falla.
    """
    # --- Comprobaciones de entrada -----------------------------------------
    if not isinstance(df, pd.DataFrame):
        print(
            "Error: el argumento 'df' debe ser un pandas.DataFrame, "
            f"se recibió {type(df).__name__}."
        )
        return None

    # bool es subclase de int en Python; lo excluimos explícitamente.
    if (
        not isinstance(umbral_categoria, int)
        or isinstance(umbral_categoria, bool)
        or umbral_categoria <= 0
    ):
        print("Error: 'umbral_categoria' debe ser un entero positivo.")
        return None

    # Aceptamos int o float para el umbral continuo, pero no bool.
    if (
        not isinstance(umbral_continua, (int, float))
        or isinstance(umbral_continua, bool)
        or not (0 <= umbral_continua <= 100)
    ):
        print("Error: 'umbral_continua' debe ser un número entre 0 y 100.")
        return None

    n_filas = len(df)

    # --- Clasificación por columna -----------------------------------------
    filas = []
    for col in df.columns:
        serie = df[col]

        # Cardinalidad = nº de valores únicos (ignorando NaN), igual que en describe_df.
        cardinalidad = serie.nunique(dropna=True)

        # Porcentaje de cardinalidad respecto al total de filas.
        porcentaje_cardinalidad = (
            0.0 if n_filas == 0 else cardinalidad / n_filas * 100
        )

        # Cascada de decisión en el orden exacto del enunciado.
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo_sugerido = "Numérica Continua"
        else:
            tipo_sugerido = "Numérica Discreta"

        filas.append(
            {"nombre_variable": col, "tipo_sugerido": tipo_sugerido}
        )

    return pd.DataFrame(filas, columns=["nombre_variable", "tipo_sugerido"])
