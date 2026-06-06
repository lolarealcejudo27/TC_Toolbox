"""
test_core_dev1.py — Tests unitarios de las funciones de Developer 1.

Cubre describe_df y tipifica_variables con los tres casos exigidos por el
enunciado (>=3 por función): caso correcto, caso límite y caso de error.

Importa desde `toolbox_ml.eda.core` (ubicación final del enunciado). Estos
tests pasarán en verde en cuanto el Scrum Master integre describe_df y
tipifica_variables (hoy en `toolbox_ml/eda/core_dev1.py`) dentro de `core.py`.

Ejecutar desde la raíz del repo:
    pytest tests/ -v
"""

import pandas as pd
import pytest

from toolbox_ml.eda.core import describe_df, tipifica_variables


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def df_mixto() -> pd.DataFrame:
    """DataFrame pequeño con tipos variados, nulos y cardinalidades controladas."""
    return pd.DataFrame(
        {
            "binaria": [0, 1, 0, 1],              # cardinalidad 2 -> Binaria
            "categorica": ["a", "b", "c", "a"],   # cardinalidad 3
            "discreta": [1, 2, 3, 4],             # cardinalidad 4 (100% card.)
            "con_nulos": [1.0, None, None, None],  # 75% nulos, cardinalidad 1
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
    # El índice debe ser el nombre de cada columna del DataFrame original.
    assert set(resultado.index) == set(df_mixto.columns)


def test_describe_df_calculos_correctos(df_mixto):
    """Comprueba nulos, valores únicos y cardinalidad en columnas concretas."""
    resultado = describe_df(df_mixto)
    # 'con_nulos': 3 de 4 valores nulos -> 75%.
    assert resultado.loc["con_nulos", "porcentaje_nulos"] == pytest.approx(75.0)
    # 'con_nulos' tiene un único valor no nulo.
    assert resultado.loc["con_nulos", "valores_unicos"] == 1
    # 'binaria' tiene 2 valores únicos sobre 4 filas -> 50% de cardinalidad.
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
    # cardinalidad 2 -> Binaria
    assert tipos["binaria"] == "Binaria"
    # cardinalidad 3 < ... no: umbral_categoria=3, card 3 NO es < 3.
    # 'categorica' tiene card 3, %card = 75 < 80 -> Numérica Discreta
    assert tipos["categorica"] == "Numérica Discreta"
    # 'discreta' card 4 >= 3, %card = 100 >= 80 -> Numérica Continua
    assert tipos["discreta"] == "Numérica Continua"
    # 'con_nulos' card 1 < 3 -> Categórica
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
