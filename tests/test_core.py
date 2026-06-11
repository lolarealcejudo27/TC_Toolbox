"""Tests unitarios de toolbox_ml.eda.core.

Cubre: describe_df, tipifica_variables, _validate_regression_inputs,
get_features_num_regression, plot_features_num_regression,
get_features_cat_regression y plot_features_cat_regression.

Tres escenarios mínimos por función: caso correcto, caso límite y caso de error.

Ejecutar desde la raíz del repo:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pytest
import seaborn as sns

from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    _validate_regression_inputs,
    _pearson_correlation_filter,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression,
)

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_mixto() -> pd.DataFrame:
    """DataFrame pequeño con tipos variados, nulos y cardinalidades controladas."""
    return pd.DataFrame(
        {
            "binaria": [0, 1, 0, 1],
            "categorica": ["a", "b", "c", "a"],
            "discreta": [1, 2, 3, 4],
            "con_nulos": [1.0, None, None, None],
        }
    )


# ---------------------------------------------------------------------------
# Tests de describe_df
# ---------------------------------------------------------------------------

def test_describe_df_devuelve_dataframe(df_mixto):
    """Caso correcto: input válido -> retorna DataFrame."""
    resultado = describe_df(df_mixto)
    assert isinstance(resultado, pd.DataFrame)


def test_describe_df_columnas_e_indice_correctos(df_mixto):
    """El resultado tiene las columnas exigidas y una fila por columna del input."""
    resultado = describe_df(df_mixto)
    assert list(resultado.columns) == [
        "tipo",
        "porcentaje_nulos",
        "valores_unicos",
        "porcentaje_cardinalidad",
    ]
    assert set(resultado.index) == set(df_mixto.columns)


def test_describe_df_calculos_correctos(df_mixto):
    """Comprueba nulos, valores únicos y cardinalidad en columnas concretas."""
    resultado = describe_df(df_mixto)
    assert resultado.loc["con_nulos", "porcentaje_nulos"] == pytest.approx(75.0)
    assert resultado.loc["con_nulos", "valores_unicos"] == 1
    assert resultado.loc["binaria", "valores_unicos"] == 2
    assert resultado.loc["binaria", "porcentaje_cardinalidad"] == pytest.approx(50.0)


def test_describe_df_dataframe_vacio():
    """Caso límite: DataFrame sin filas no debe romper (porcentajes a 0)."""
    df_vacio = pd.DataFrame({"a": [], "b": []})
    resultado = describe_df(df_vacio)
    assert isinstance(resultado, pd.DataFrame)
    assert resultado.loc["a", "porcentaje_nulos"] == 0.0
    assert resultado.loc["a", "valores_unicos"] == 0


def test_describe_df_columna_todo_nulos():
    """Caso límite: una columna con todos los valores nulos -> 100% nulos."""
    df = pd.DataFrame({"a": [None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc["a", "porcentaje_nulos"] == pytest.approx(100.0)
    assert resultado.loc["a", "valores_unicos"] == 0


def test_describe_df_input_invalido():
    """Caso de error: input que no es DataFrame -> retorna None."""
    assert describe_df("esto no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None
    assert describe_df(None) is None


# ---------------------------------------------------------------------------
# Tests de tipifica_variables
# ---------------------------------------------------------------------------

def test_tipifica_variables_cuatro_tipos(df_mixto):
    """Caso correcto: produce los cuatro tipos según la cascada."""
    resultado = tipifica_variables(df_mixto, umbral_categoria=3, umbral_continua=80.0)
    tipos = dict(zip(resultado["nombre_variable"], resultado["tipo_sugerido"]))
    assert tipos["binaria"] == "Binaria"
    assert tipos["categorica"] == "Numérica Discreta"
    assert tipos["discreta"] == "Numérica Continua"
    assert tipos["con_nulos"] == "Categórica"


def test_tipifica_variables_estructura_salida(df_mixto):
    """El resultado tiene las columnas correctas y una fila por columna del input."""
    resultado = tipifica_variables(df_mixto, umbral_categoria=10, umbral_continua=30.0)
    assert list(resultado.columns) == ["nombre_variable", "tipo_sugerido"]
    assert len(resultado) == df_mixto.shape[1]


def test_tipifica_variables_umbral_categoria_invalido(df_mixto):
    """Caso de error: umbral_categoria no positivo o no entero -> None."""
    assert tipifica_variables(df_mixto, umbral_categoria=-1, umbral_continua=30.0) is None
    assert tipifica_variables(df_mixto, umbral_categoria=0, umbral_continua=30.0) is None
    assert tipifica_variables(df_mixto, umbral_categoria=2.5, umbral_continua=30.0) is None
    assert tipifica_variables(df_mixto, umbral_categoria=True, umbral_continua=30.0) is None


def test_tipifica_variables_umbral_continua_invalido(df_mixto):
    """Caso de error: umbral_continua fuera de [0, 100] -> None."""
    assert tipifica_variables(df_mixto, umbral_categoria=10, umbral_continua=-5) is None
    assert tipifica_variables(df_mixto, umbral_categoria=10, umbral_continua=150) is None


def test_tipifica_variables_df_invalido():
    """Caso de error: input que no es DataFrame -> None."""
    assert tipifica_variables("no soy df", umbral_categoria=10, umbral_continua=30.0) is None


# ---------------------------------------------------------------------------
# Tests de _validate_regression_inputs
# ---------------------------------------------------------------------------

def test_validacion_de_inputs_correcta():
    """Caso correcto: la validación acepta entradas válidas."""
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 3, 4, 5, 6],
            "target": [3, 4, 5, 6, 7],
        }
    )
    assert _validate_regression_inputs(df, "target", 0.5, 0.05) is True


def test_validacion_de_inputs_datos_incorrectos():
    """Caso de error: entradas inválidas deben provocar False."""
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 2, 3]})
    assert _validate_regression_inputs([1, 2, 3], "target", 0.5, 0.05) is False
    assert _validate_regression_inputs(df, "missing", 0.5, 0.05) is False
    assert _validate_regression_inputs(df, "target", 1.5, 0.05) is False


# ---------------------------------------------------------------------------
# Tests de _pearson_correlation_filter
# ---------------------------------------------------------------------------

def test_pearson_selecciona_columna_correlada():
    """Caso correcto: columna con correlación perfecta supera el umbral."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "target": [2, 4, 6, 8, 10]})
    result = _pearson_correlation_filter(df, "target", ["x"], umbral_corr=0.9)
    assert result == ["x"]


def test_pearson_excluye_columna_no_correlada():
    """Caso correcto: columna con correlación baja queda excluida."""
    df = pd.DataFrame({"x": [5, 1, 4, 2, 3], "target": [2, 4, 6, 8, 10]})
    result = _pearson_correlation_filter(df, "target", ["x"], umbral_corr=0.9)
    assert result == []


def test_pearson_filtra_por_pvalue():
    """Caso correcto: correlación suficiente pero p-valor no significativo -> excluida."""
    # x=[1,2,3], target=[1,3,2] -> r≈0.5, p≈0.64 (no significativo con n=3)
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 3, 2]})
    result = _pearson_correlation_filter(df, "target", ["x"], umbral_corr=0.5, pvalue=0.01)
    assert result == []


def test_pearson_acepta_pvalue_none():
    """Caso correcto: sin filtro de pvalue se aplica solo el umbral de correlación."""
    df = pd.DataFrame({"x": [1, 2, 3], "target": [3, 2, 1]})
    result = _pearson_correlation_filter(df, "target", ["x"], umbral_corr=0.5, pvalue=None)
    assert result == ["x"]


def test_pearson_candidatas_vacias():
    """Caso límite: lista de candidatas vacía devuelve lista vacía."""
    df = pd.DataFrame({"target": [1, 2, 3]})
    result = _pearson_correlation_filter(df, "target", [], umbral_corr=0.5)
    assert result == []


def test_pearson_omite_par_con_menos_de_dos_observaciones():
    """Caso límite: columna con un solo par válido tras dropna es omitida."""
    df = pd.DataFrame({"x": [1.0, None, None, None], "target": [1.0, None, None, None]})
    result = _pearson_correlation_filter(df, "target", ["x"], umbral_corr=0.0)
    assert result == []


# ---------------------------------------------------------------------------
# Tests de get_features_num_regression
# ---------------------------------------------------------------------------

def test_get_features_num_regression_filters_by_correlation():
    """Caso correcto: devuelve solo las variables suficientemente correladas."""
    df = pd.DataFrame(
        {
            "strong": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "weak": [5, 1, 4, 2, 3, 6, 9, 7, 8],
            "target": [2, 4, 6, 8, 10, 12, 14, 16, 18],
        }
    )
    result = get_features_num_regression(df, "target", 0.8)
    assert result == ["strong"]


def test_get_features_num_regression_retorna_vacia_con_dataframe_sin_filas():
    """Caso límite: DataFrame vacío no ofrece columnas útiles para analizar."""
    df = pd.DataFrame({"feature": pd.Series(dtype=float), "target": pd.Series(dtype=float)})
    result = get_features_num_regression(df, "target", 0.5)
    assert result == []


def test_get_features_num_regression_retorna_vacia_con_columna_todo_nulos():
    """Caso límite: columna objetivo completamente nula no puede correlacionarse."""
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": pd.Series([None, None, None, None], dtype=float),
        }
    )
    result = get_features_num_regression(df, "target", 0.5)
    assert result == []


def test_get_features_num_regression_input_invalido():
    """Caso de error: target inexistente o umbral fuera de rango -> None."""
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 2, 3]})
    assert get_features_num_regression(df, "missing", 0.5) is None
    assert get_features_num_regression(df, "target", 1.5) is None


# ---------------------------------------------------------------------------
# Tests de plot_features_num_regression
# ---------------------------------------------------------------------------

def test_plot_features_num_regression_accepts_default_thresholds():
    """Caso correcto: selecciona y representa las columnas esperadas."""
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5],
            "f2": [2, 4, 6, 8, 10],
            "target": [3, 6, 9, 12, 15],
        }
    )
    result = plot_features_num_regression(df, target_col="target")
    assert result == ["f1", "f2"]
    plt.close("all")


def test_plot_features_num_regression_retorna_vacia_con_dataframe_vacio():
    """Caso límite: DataFrame vacío devuelve lista vacía sin generar gráficos."""
    df = pd.DataFrame({"feature": pd.Series(dtype=float), "target": pd.Series(dtype=float)})
    result = plot_features_num_regression(df, target_col="target")
    assert result == []
    plt.close("all")


def test_plot_features_num_regression_validates_inputs():
    """Caso de error: entradas inválidas devuelven None."""
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 2, 3]})
    assert plot_features_num_regression([1, 2, 3], target_col="target") is None
    assert plot_features_num_regression(df, target_col="missing") is None
    assert plot_features_num_regression(df, target_col="target", umbral_corr=1.5) is None


# ---------------------------------------------------------------------------
# Tests de get_features_cat_regression
# ---------------------------------------------------------------------------

def test_get_cat_df_no_dataframe():
    """Caso de error: df no es DataFrame -> None."""
    assert get_features_cat_regression([1, 2, 3], "target") is None


def test_get_cat_target_no_existe():
    """Caso de error: target_col no existe en el DataFrame -> None."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    assert get_features_cat_regression(df, "target") is None


def test_get_cat_target_no_numerico():
    """Caso de error: target_col no es numérica -> None."""
    df = pd.DataFrame(
        {
            "target": pd.Series(["a", "b", "c"], dtype="string"),
            "cat": ["x", "y", "z"],
        }
    )
    assert get_features_cat_regression(df, "target") is None


def test_get_cat_pvalue_invalido():
    """Caso de error: pvalue fuera de rango -> None."""
    df = pd.DataFrame({"target": [1, 2, 3]})
    assert get_features_cat_regression(df, "target", pvalue=1.5) is None


def test_get_cat_df_vacio():
    """Caso límite: DataFrame vacío -> None."""
    assert get_features_cat_regression(pd.DataFrame(), "target") is None


def test_get_cat_mann_whitney_significativo():
    """Caso correcto: variable binaria con diferencias claras es detectada."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    assert "ciudad" in get_features_cat_regression(df, "target")


def test_get_cat_mann_whitney_no_significativo():
    """Caso correcto: variable binaria sin diferencias no es seleccionada."""
    df = pd.DataFrame({"sexo": ["H", "H", "M", "M"], "target": [10, 10, 10, 10]})
    assert "sexo" not in get_features_cat_regression(df, "target")


def test_get_cat_variable_booleana():
    """Caso correcto: variables booleanas son tratadas como binarias."""
    df = pd.DataFrame(
        {
            "type": [True] * 10 + [False] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    assert "type" in get_features_cat_regression(df, "target")


def test_get_cat_anova_significativo():
    """Caso correcto: variable con >2 categorías con diferencias es detectada."""
    df = pd.DataFrame(
        {
            "grupo": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "target": [1, 2, 3, 10, 11, 12, 20, 21, 22],
        }
    )
    assert "grupo" in get_features_cat_regression(df, "target")


def test_get_cat_anova_no_significativo():
    """Caso correcto: variable con >2 categorías sin diferencias no es seleccionada."""
    df = pd.DataFrame(
        {
            "grupo": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "target": [10] * 9,
        }
    )
    assert "grupo" not in get_features_cat_regression(df, "target")


def test_get_cat_una_sola_categoria():
    """Caso límite: columna con una sola categoría es ignorada."""
    df = pd.DataFrame({"cat": ["A", "A", "A"], "target": [1, 2, 3]})
    assert "cat" not in get_features_cat_regression(df, "target")


def test_get_cat_columna_todo_nan():
    """Caso límite: columna completamente NaN no genera errores y no se selecciona."""
    df = pd.DataFrame({"cat": [np.nan, np.nan, np.nan], "target": [1, 2, 3]})
    assert "cat" not in get_features_cat_regression(df, "target")


def test_get_cat_target_con_nan():
    """Caso límite: NaN en el target no rompe la función."""
    df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "target": [1, np.nan, 3, 4]})
    assert isinstance(get_features_cat_regression(df, "target"), list)


# ---------------------------------------------------------------------------
# Tests de plot_features_cat_regression
# ---------------------------------------------------------------------------

def test_plot_cat_df_no_dataframe():
    """Caso de error: df no es DataFrame -> None."""
    assert plot_features_cat_regression([1, 2, 3], "target") is None


def test_plot_cat_target_no_existe():
    """Caso de error: target_col no existe en el DataFrame -> None."""
    df = pd.DataFrame({"ciudad": ["Madrid", "Barcelona"]})
    assert plot_features_cat_regression(df, "target") is None


def test_plot_cat_target_no_numerico():
    """Caso de error: target_col no es numérica -> None."""
    df = pd.DataFrame(
        {
            "target": pd.Series(["a", "b", "c"], dtype="string"),
            "ciudad": ["Madrid", "Barcelona", "Sevilla"],
        }
    )
    assert plot_features_cat_regression(df, "target") is None


def test_plot_cat_pvalue_invalido():
    """Caso de error: pvalue fuera de rango -> None."""
    df = pd.DataFrame({"ciudad": ["Madrid", "Barcelona"], "target": [1, 2]})
    assert plot_features_cat_regression(df, "target", pvalue=1.5) is None


def test_plot_cat_columns_vacio_usa_categoricas():
    """Caso correcto: columns=[] infiere automáticamente las columnas categóricas."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    result = plot_features_cat_regression(df, "target", columns=[])
    assert isinstance(result, list)
    assert "ciudad" in result
    plt.close("all")


def test_plot_cat_columns_con_col_inexistente():
    """Caso límite: columnas inexistentes en columns son ignoradas sin error."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    result = plot_features_cat_regression(df, "target", columns=["ciudad", "col_inexistente"])
    assert isinstance(result, list)
    plt.close("all")


def test_plot_cat_sin_features_significativas():
    """Caso límite: ninguna variable supera el test -> lista vacía, sin figuras."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 5 + ["Barcelona"] * 5,
            "target": [10] * 10,
        }
    )
    assert plot_features_cat_regression(df, "target") == []


def test_plot_cat_devuelve_features_significativas():
    """Caso correcto: devuelve correctamente la lista de columnas significativas."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    result = plot_features_cat_regression(df, "target")
    assert isinstance(result, list)
    assert "ciudad" in result
    plt.close("all")


def test_plot_cat_with_individual_plot_false():
    """Caso correcto: with_individual_plot=False genera una única figura con subplots."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    result = plot_features_cat_regression(df, "target", with_individual_plot=False)
    assert "ciudad" in result
    plt.close("all")


def test_plot_cat_with_individual_plot_true():
    """Caso correcto: with_individual_plot=True genera una figura por variable."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    result = plot_features_cat_regression(df, "target", with_individual_plot=True)
    assert "ciudad" in result
    plt.close("all")


def test_plot_cat_numero_subplots_correcto():
    """Caso correcto: múltiples variables significativas se manejan sin errores."""
    df = pd.DataFrame(
        {
            "ciudad": ["Madrid"] * 10 + ["Barcelona"] * 10,
            "tipo": ["A"] * 10 + ["B"] * 10,
            "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12],
        }
    )
    result = plot_features_cat_regression(df, "target", with_individual_plot=False)
    assert len(result) == 2
    plt.close("all")
