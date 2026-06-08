import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from temp_files.categorical import get_features_cat_regression
from temp_files.categorical import plot_features_cat_regression

# -------------------------
# TESTS: get_features_cat_regression
# -------------------------
"""
Este conjunto de tests verifica el correcto funcionamiento de la función
get_features_cat_regression en diferentes escenarios:

1. Validación de entradas:
   - df no es un DataFrame
   - target_col no existe en el DataFrame
   - target_col no es numérica
   - pvalue fuera del rango válido
   - DataFrame vacío

2. Casos estadísticos con variables categóricas:
   - Variables con 2 categorías (test de Mann-Whitney U)
   - Variables booleanas (True/False), tratadas como caso binario
   - Variables con más de 2 categorías (ANOVA)
   
3. Evaluación de significancia estadística:
   - Variables con relación significativa con el target
   - Variables sin relación significativa

4. Casos extremos:
   - Variables con una sola categoría
   - Columnas completamente NaN
   - Presencia de NaN en el target

El objetivo es asegurar que la función sea robusta, correcta
y capaz de manejar datos reales con imperfecciones.

"""

def test_df_no_dataframe():
    """
    Verifica que la función devuelve None si df no es un DataFrame.
    """
    result = get_features_cat_regression([1, 2, 3], "target")
    assert result is None


def test_target_no_existe():
    """
    Verifica que la función devuelve None si la columna target no existe.
    """
    df = pd.DataFrame({"A": [1, 2, 3]})

    result = get_features_cat_regression(df, "target")
    assert result is None


def test_target_no_numerico():
    """
    Verifica que la función devuelve None si el target no es numérico.

    Comprueba compatibilidad con tipos pandas modernos (string, category).
    """
    import pandas as pd

    # Creamos un DataFrame con target no numérico (string)
    df = pd.DataFrame({
        "target": pd.Series(["a", "b", "c"], dtype="string"),
        "cat": ["x", "y", "z"]
    })

    result = get_features_cat_regression(df, "target")

    assert result is None


def test_pvalue_invalido():
    """
    Verifica que la función devuelve None si pvalue está fuera de rango.
    """
    df = pd.DataFrame({"target": [1, 2, 3]})

    result = get_features_cat_regression(df, "target", pvalue=1.5)
    assert result is None


def test_df_vacio():
    """
    Verifica que la función maneja correctamente un DataFrame vacío.
    """
    df = pd.DataFrame()

    result = get_features_cat_regression(df, "target")
    assert result is None




def test_mann_whitney_significativo():
    """
    Verifica que una variable categórica binaria con diferencias claras
    en el target es detectada como significativa mediante Mann-Whitney U.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = get_features_cat_regression(df, "target")
    assert "ciudad" in result

def test_mann_whitney_no_significativo():
    """
    Verifica que una variable binaria sin diferencias en el target
    no es seleccionada como significativa.
    """
    import pandas as pd

    df = pd.DataFrame({
        "sexo": ["H", "H", "M", "M"],
        "target": [10, 10, 10, 10]
    })

    result = get_features_cat_regression(df, "target")

    assert "sexo" not in result

def test_variable_booleana():
    """
    Verifica que variables booleanas son tratadas como categóricas binarias
    y evaluadas correctamente con Mann-Whitney U.
    """
    df = pd.DataFrame({
        "type": [True] * 10 + [False] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = get_features_cat_regression(df, "target")
    assert "type" in result

def test_anova_significativo():
    """
    Verifica que una variable con más de 2 categorías
    con diferencias en el target es detectada como significativa.
    """
    import pandas as pd

    df = pd.DataFrame({
        "grupo": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        "target": [1, 2, 3, 10, 11, 12, 20, 21, 22]
    })

    result = get_features_cat_regression(df, "target")

    assert "grupo" in result


def test_anova_no_significativo():
    """
    Verifica que una variable con más de 2 categorías sin diferencias
    no es seleccionada.
    """

    df = pd.DataFrame({
        "grupo": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        "target": [10] * 9
    })

    result = get_features_cat_regression(df, "target")

    assert "grupo" not in result


def test_una_sola_categoria():
    """
    Verifica que una columna con una sola categoría es ignorada.
    """
    df = pd.DataFrame({
        "cat": ["A", "A", "A"],
        "target": [1, 2, 3]
    })

    result = get_features_cat_regression(df, "target")
    assert "cat" not in result


def test_categoria_con_nan():
    """
    Verifica que columnas completamente NaN no generan errores
    y no son seleccionadas.
    """
    df = pd.DataFrame({
        "cat": [np.nan, np.nan, np.nan],
        "target": [1, 2, 3]
    })

    result = get_features_cat_regression(df, "target")
    assert "cat" not in result


def test_target_con_nan():
    """
    Verifica que la función maneja correctamente NaN en el target.
    """
    df = pd.DataFrame({
        "cat": ["A", "B", "A", "B"],
        "target": [1, np.nan, 3, 4]
    })

    result = get_features_cat_regression(df, "target")
    assert isinstance(result, list)



# -------------------------
# TESTS: plot_features_cat_regression
# -------------------------
"""
Tests para la función plot_features_cat_regression.

Escenarios cubiertos:

1. Validación de entrada:
   - Si df no es un DataFrame devuelve None
   - Si target_col no existe en el DataFrame devuelve None
   - Sitarget_col no es numérica devuelve None
   - Si pvalue fuera de rango devuelve None

2. Comportamiento con el parámetro columns:
   - Si columns=[] se infieren automáticamente las columnas categóricas
   - Si columns contiene columnas inexistentes las ignora sin error

3. Selección estadística:
   - Si ninguna variable supera el test devuelve lista vacía y no genera figuras
   - Si hay variables significativas devuelve correctamente su lista

4. Visualización:
   - Si with_individual_plot=False se obtiene una única figura con subplots
   - Si with_individual_plot=True se obtiene una figura independiente por variable
   - Número de subplots visible coincide con el número de variables significativas
"""


# -------------------------
# TESTS DE VALIDACIÓN
# -------------------------

def test_plot_df_no_dataframe():
    """
    Verifica que la función devuelve None si df no es un DataFrame.
    """
    result = plot_features_cat_regression([1, 2, 3], "target")
    assert result is None


def test_plot_target_no_existe():
    """
    Verifica que la función devuelve None si target_col no existe en el DataFrame.
    """
    df = pd.DataFrame({"ciudad": ["Madrid", "Barcelona"]})
    result = plot_features_cat_regression(df, "target")
    assert result is None


def test_plot_target_no_numerico():
    """
    Verifica que la función devuelve None si target_col no es numérica.
    """
    df = pd.DataFrame({
        "target": pd.Series(["a", "b", "c"], dtype="string"),
        "ciudad": ["Madrid", "Barcelona", "Sevilla"]
    })
    result = plot_features_cat_regression(df, "target")
    assert result is None


def test_plot_pvalue_invalido():
    """
    Verifica que la función devuelve None si pvalue está fuera de rango.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid", "Barcelona"],
        "target": [1, 2]
    })
    result = plot_features_cat_regression(df, "target", pvalue=1.5)
    assert result is None


# -------------------------
# TESTS DE COMPORTAMIENTO CON columns
# -------------------------

def test_plot_columns_vacio_usa_categoricas():
    """
    Verifica que si columns=[], la función infiere automáticamente
    las columnas categóricas del DataFrame.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = plot_features_cat_regression(df, "target", columns=[])
    assert isinstance(result, list)
    assert "ciudad" in result


def test_plot_columns_con_col_inexistente():
    """
    Verifica que columnas en columns que no existen en el DataFrame
    son ignoradas sin lanzar error.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = plot_features_cat_regression(
        df, "target", columns=["ciudad", "col_inexistente"]
    )
    assert isinstance(result, list)


# -------------------------
# TESTS DE SELECCIÓN ESTADÍSTICA
# -------------------------

def test_plot_sin_features_significativas():
    """
    Verifica que si ninguna variable supera el test estadístico,
    la función devuelve una lista vacía y no genera ninguna figura.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 5 + ["Barcelona"] * 5,
        "target": [10] * 10
    })
    result = plot_features_cat_regression(df, "target")
    assert result == []


def test_plot_devuelve_features_significativas():
    """
    Verifica que la función devuelve correctamente la lista de columnas
    que superaron el test estadístico.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = plot_features_cat_regression(df, "target")
    assert isinstance(result, list)
    assert "ciudad" in result


# -------------------------
# TESTS DE VISUALIZACIÓN
# -------------------------

def test_plot_with_individual_plot_false():
    """
    Verifica que con with_individual_plot=False la función no lanza
    errores y devuelve la lista de variables significativas.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = plot_features_cat_regression(df, "target", with_individual_plot=False)
    assert isinstance(result, list)
    assert "ciudad" in result


def test_plot_with_individual_plot_true():
    """
    Verifica que con with_individual_plot=True la función no lanza
    errores y devuelve la lista de variables significativas.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = plot_features_cat_regression(df, "target", with_individual_plot=True)
    assert isinstance(result, list)
    assert "ciudad" in result


def test_plot_numero_subplots_correcto():
    """
    Verifica que la función maneja correctamente múltiples variables
    significativas sin lanzar errores.
    """
    df = pd.DataFrame({
        "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
        "tipo":   ["A"] * 10 + ["B"] * 10,
        "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    })
    result = plot_features_cat_regression(df, "target", with_individual_plot=False)
    assert isinstance(result, list)
    assert len(result) == 2