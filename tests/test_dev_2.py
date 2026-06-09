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

def test_get_features_num_regression_datos_incorrectos():
    """Caso de error: entradas inválidas deben provocar `None`.

    Se comprueba que un objeto que no es DataFrame, una columna inexistente y un
    umbral fuera de rango sean rechazados de forma explícita.
    """
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 2, 3]})

    assert module.get_features_num_regression([1, 2, 3], "target", 0.5) is None
    assert module.get_features_num_regression(df, "missing", 0.5) is None
    assert module.get_features_num_regression(df, "target", 1.5) is None


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
