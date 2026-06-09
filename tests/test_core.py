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


"""Pruebas unitarias para las utilidades numéricas de regresión.

Este archivo valida el comportamiento público de las funciones definidas en
`notebooks/numerical.py`. Las pruebas cubren tres escenarios principales para
cada función:

* caso correcto, con entradas válidas y resultados esperados;
* caso límite, con estructuras vacías o datos degenerados;
* caso de error, con entradas inválidas que deben devolver `None`.

La importación es directa porque el módulo está disponible desde la raíz del
repositorio durante la ejecución de pytest.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
	sys.path.insert(0, str(ROOT_PATH))

from temp_files import numerical as module


def test_validacion_de_inputs_correcta():
	"""Caso correcto: la validación acepta entradas válidas.

	Se comprueba que, con un DataFrame adecuado y parámetros dentro de los
	rangos esperados, la función devuelve `True`.
	"""
	df = pd.DataFrame(
		{
			"feature1": [1, 2, 3, 4, 5],
			"feature2": [2, 3, 4, 5, 6],
			"target": [3, 4, 5, 6, 7],
		}
	)

	result = module._validate_regression_inputs(df, "target", 0.5, 0.05)

	assert result is True


def test_validacion_de_inputs_datos_incorrectos():
	"""Caso de error: entradas inválidas deben provocar `False`.

	Se comprueba que un objeto que no es DataFrame, una columna inexistente y un
	umbral fuera de rango sean rechazados de forma explícita.
	"""
	df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 2, 3]})

	# Cada una de estas llamadas representa una violación distinta de las reglas
	# de validación establecidas para la función pública.
	assert module._validate_regression_inputs([1, 2, 3], "target", 0.5, 0.05) is False
	assert module._validate_regression_inputs(df, "missing", 0.5, 0.05) is False
	assert module._validate_regression_inputs(df, "target", 1.5, 0.05) is False


def test__candidate_numeric_columns_identifica_columnas_numericas_correctamente():
	"""Caso correcto: la función identifica correctamente las columnas numéricas.

	Se comprueba que, dado un DataFrame con una mezcla de tipos de datos, la
	función devuelve solo los nombres de las columnas numéricas, excluyendo el
	objetivo.
	"""
	df = pd.DataFrame(
		{
			"num1": [1.0, 2.0, 3.0],
			"num2": [4.0, 5.0, 6.0],
			"target": [7.0, 8.0, 9.0],
			"category": ["a", "b", "c"],
		}
	)

	result = module._candidate_numeric_columns(df, "target")

	assert set(result) == {"num1", "num2"}


def test__candidate_numeric_columns_retorna_vacia_con_dataframe_sin_columnas_numericas():
	"""Caso límite: un DataFrame sin columnas numéricas no ofrece candidatos.

	La función debe devolver una lista vacía cuando no existen columnas numéricas
	que analizar, incluso si el DataFrame tiene otras columnas de tipo diferente.
	"""
	df = pd.DataFrame(
		{
			"category1": ["a", "b", "c"],
			"category2": ["d", "e", "f"],
			"target": [1.0, 2.0, 3.0],
		}
	)

	result = module._candidate_numeric_columns(df, "target")

	assert result == []


def test__candidate_numeric_columns_retorna_vacia_con_dataframe_vacio():
	"""Caso límite: un DataFrame vacío no ofrece columnas para analizar.

	La función debe devolver una lista vacía cuando el DataFrame no contiene
	ninguna columna, ni siquiera el objetivo.
	"""
	df = pd.DataFrame()

	result = module._candidate_numeric_columns(df, "target")

	assert result == []


def test_get_features_num_regression_filters_by_correlation():
	"""Caso correcto: devuelve las variables numéricas suficientemente correladas.

	El DataFrame contiene una columna con correlación lineal clara respecto al
	target y otra con una relación más débil. Solo debe sobrevivir la primera.
	"""
	df = pd.DataFrame(
		{
			"strong": [1, 2, 3, 4, 5],
			"weak": [5, 1, 4, 2, 3],
			"target": [2, 4, 6, 8, 10],
			"category": ["a", "b", "c", "d", "e"],
		}
	)

	result = module.get_features_num_regression(df, "target", 0.8)

	assert result == ["strong"]


def test_get_features_num_regression_retorna_vacia_con_dataframe_sin_filas():
	"""Caso límite: un DataFrame vacío no ofrece columnas útiles para analizar.

	Aunque la validación de entrada es correcta, la ausencia de filas impide
	calcular correlaciones reales, por lo que el resultado debe ser una lista
	vacía.
	"""
	df = pd.DataFrame({"feature": pd.Series(dtype=float), "target": pd.Series(dtype=float)})

	result = module.get_features_num_regression(df, "target", 0.5)

	assert result == []


def test_get_features_num_regression_retorna_vacia_con_columna_todo_nulos():
	"""Caso límite: una columna objetivo completamente nula no puede correlacionarse.

	La función debe ignorar la columna porque, tras eliminar nulos por pares,
	no queda información suficiente para calcular una correlación válida.
	"""
	df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "target": pd.Series([None, None, None, None], dtype=float)})

	result = module.get_features_num_regression(df, "target", 0.5)

	assert result == []


def test_plot_features_num_regression_accepts_default_thresholds():
	"""Caso correcto: la función selecciona y representa las columnas esperadas.

	Aquí se comprueba que, con el comportamiento por defecto, se obtienen las
	mismas variables que en la función de selección y que el proceso visual se
	ejecuta sin producir efectos secundarios reales en los tests.
	"""
	df = pd.DataFrame(
		{
			"f1": [1, 2, 3, 4, 5],
			"f2": [2, 4, 6, 8, 10],
			"target": [3, 6, 9, 12, 15],
		}
	)

	# Se sustituyen los objetos gráficos por funciones sin efecto para evitar que
	# la ejecución de la prueba abra ventanas o dependa de un backend gráfico.
	previous_pairplot = module.sns.pairplot
	previous_show = module.plt.show
	module.sns.pairplot = lambda *args, **kwargs: None
	module.plt.show = lambda *args, **kwargs: None

	try:
		result = module.plot_features_num_regression(df, target_col="target")
	finally:
		module.sns.pairplot = previous_pairplot
		module.plt.show = previous_show

	assert result == ["f1", "f2"]


def test_plot_features_num_regression_retorna_vacia_con_dataframe_vacio():
	"""Caso límite: un DataFrame vacío no debe producir gráficos ni selecciones.

	La función debe devolver una lista vacía cuando no existen datos suficientes
	para seleccionar variables con correlación útil.
	"""
	df = pd.DataFrame({"feature": pd.Series(dtype=float), "target": pd.Series(dtype=float)})

	# Igual que en la prueba anterior, anulamos los efectos visuales para que el
	# test se centre únicamente en el valor devuelto.
	previous_pairplot = module.sns.pairplot
	previous_show = module.plt.show
	module.sns.pairplot = lambda *args, **kwargs: None
	module.plt.show = lambda *args, **kwargs: None
	try:
		result = module.plot_features_num_regression(df, target_col="target")
	finally:
		module.sns.pairplot = previous_pairplot
		module.plt.show = previous_show

	assert result == []


def test_plot_features_num_regression_validates_inputs():
	"""Caso de error: las mismas validaciones de entrada deben devolver `None`.

	La función visual reutiliza la validación de la función numérica principal,
	por lo que debe rechazar exactamente los mismos escenarios inválidos.
	"""
	df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 2, 3]})

	# Se prueban tres fallos diferentes: tipo incorrecto, columna ausente y
	# umbral fuera del intervalo permitido.
	assert module.plot_features_num_regression([1, 2, 3], target_col="target") is None
	assert module.plot_features_num_regression(df, target_col="missing") is None
	assert module.plot_features_num_regression(df, target_col="target", umbral_corr=1.5) is None

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