"""Utilidades para selección de variables numéricas y visualización.

Este módulo proporciona dos ayudas públicas orientadas a problemas de
regresión:

`get_features_num_regression`
	Selecciona las columnas numéricas que están suficientemente correladas
	con una columna objetivo también numérica.

`plot_features_num_regression`
	Reutiliza la misma lógica de selección y representa pairplots para las
	columnas elegidas en grupos cuando el conjunto de variables es grande.

La implementación es intencionadamente pequeña y directa: la validación de
entradas se realiza al principio, la columna objetivo debe ser numérica y el
filtrado se basa en la correlación de Pearson y, de forma opcional, en un
umbral de p-valor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from toolbox_ml.eda.core_dev1 import tipifica_variables


def _validate_regression_inputs(
	df: pd.DataFrame,
	target_col: str,
	umbral_corr: float,
	pvalue: float = None,
) -> bool:
	"""Valida las entradas comunes usadas por las utilidades de regresión.

	Parámetros
	----------
	df:
		DataFrame que contiene las variables candidatas y la columna objetivo.
	target_col:
		Nombre de la columna objetivo. Debe existir en `df` y ser numérica.
	umbral_corr:
		Correlación de Pearson mínima en valor absoluto para conservar una
		columna.
	pvalue:
		Umbral opcional de p-valor. Si se indica, solo se aceptan correlaciones
		estadísticamente significativas.

	Devuelve
	--------
	bool
		`True` cuando todas las entradas cumplen las restricciones esperadas;
		`False` en caso contrario. La función imprime un motivo breve antes de
		devolver `False`.
	"""
	# El selector solo tiene sentido si la entrada es un DataFrame.
	if not isinstance(df, pd.DataFrame):
		print("df debe ser un pd.DataFrame")
		return False

	# La columna objetivo debe existir porque todas las operaciones posteriores
	# dependen de ella.
	if target_col not in df.columns:
		print(f"target_col '{target_col}' no existe en df")
		return False

	# La correlación de Pearson solo es válida para objetivos numéricos.
	if not pd.api.types.is_numeric_dtype(df[target_col]):
		print(f"target_col '{target_col}' debe ser una columna numérica")
		return False

	# Aceptamos enteros y flotantes para que valores por defecto como 0 sean
	# válidos, pero excluimos booleanos porque Python los considera enteros.
	if not isinstance(umbral_corr, (int, float)) or isinstance(umbral_corr, bool) or not 0 <= umbral_corr <= 1:
		print("umbral_corr debe ser un float entre 0 y 1")
		return False

	# El filtro de p-valor es opcional. Si se usa, también debe ser numérico y
	# estar en el mismo intervalo cerrado.
	if pvalue is not None and (
		not isinstance(pvalue, (int, float)) or isinstance(pvalue, bool) or not 0 <= pvalue <= 1
	):
		print("pvalue, si no es None, debe ser un float entre 0 y 1")
		return False

	return True



def _pearson_correlation_filter(
	df: pd.DataFrame,
	target_col: str,
	candidate_columns: list[str],
	umbral_corr: float,
	pvalue: float = None,
) -> list[str]:
	"""Filtra columnas candidatas usando correlación de Pearson y p-valor opcional.

	Para cada columna, la función elimina de forma conjunta los valores nulos
	con respecto al objetivo, calcula la correlación de Pearson y su p-valor, y
	conserva la columna solo si se cumplen ambas condiciones:

	* `abs(r) >= umbral_corr`
	* `pvalue is None` o `p_val < pvalue`

	La salida mantiene el orden original de las columnas candidatas.
	"""
	selected_columns: list[str] = []

	# Filtramos cada columna de forma independiente para que los nulos o un
	# fallo en una candidata no afecten al resto de la lista.
	for column in candidate_columns:
		# Pearson requiere pares alineados y sin nulos. Eliminar filas por pares
		# evita contaminar el cálculo con datos ausentes.
		pair = df[[column, target_col]].dropna()
		if len(pair) < 2:
			continue

		# La ayuda de scipy devuelve tanto el coeficiente de correlación como el
		# p-valor del contraste de hipótesis de correlación nula.
		try:
			corr, p_val = pearsonr(pair[column], pair[target_col])
		except Exception:
			# Una serie mal formada o constante puede provocar una excepción en
			# scipy; lo más seguro es omitir esa variable y continuar.
			continue

		# Los NaN suelen indicar una correlación inválida o degenerada.
		if pd.isna(corr) or pd.isna(p_val):
			continue

		# Aplicamos el umbral de correlación absoluta pedido por el usuario.
		if abs(corr) < umbral_corr:
			continue

		# Aplicamos el filtro opcional de significación estadística cuando se ha
		# solicitado.
		if pvalue is not None and p_val >= pvalue:
			continue

		# Conservamos la variable solo cuando cumple todas las condiciones.
		selected_columns.append(column)

	return selected_columns


def get_features_num_regression(
	df: pd.DataFrame,
	target_col: str,
	umbral_corr: float,
	pvalue: float = None,
) -> list:
	"""Devuelve predictores numéricos correlados con un objetivo numérico.

	Esta ayuda inspecciona todas las columnas numéricas de `df`, excluye
	`target_col` y conserva solo aquellas variables cuya correlación de Pearson
	con el objetivo alcanza el umbral absoluto `umbral_corr`.

	Cuando se proporciona `pvalue`, la función añade un filtro de significación y
	retiene únicamente las variables cuyo p-valor en el test de Pearson es menor
	que el umbral indicado.

	Parámetros
	----------
	df:
		DataFrame de entrada que contiene el objetivo y las variables numéricas
		candidatas.
	target_col:
		Nombre de la columna de respuesta numérica.
	umbral_corr:
		Umbral de correlación absoluta en el rango [0, 1].
	pvalue:
		Umbral opcional de p-valor en el rango [0, 1].

	Devuelve
	--------
	list
		Lista con los nombres de las columnas que cumplen las reglas de
		correlación.
		Devuelve `None` cuando falla la validación de entrada.
	"""

	if not _validate_regression_inputs(df, target_col, umbral_corr, pvalue):
		return None

	# Obtenemos la tipificación sugerida por la función compartida. Usamos
	# umbrales por defecto razonables: `umbral_categoria=3` y
	# `umbral_continua=50`. Solo aceptamos las variables marcadas como
	# "Numérica ..." y que además tengan un dtype numérico en el DataFrame.
	tip = tipifica_variables(df, umbral_categoria=3, umbral_continua=50)
	if tip is None:
		candidate_columns = []
	else:
		candidate_columns = [
			col
			for col in tip.loc[tip["tipo_sugerido"].str.startswith("Numérica"), "nombre_variable"].tolist()
			if col in df.columns and col != target_col and pd.api.types.is_numeric_dtype(df[col])
		]
	# Delegamos el cálculo y el filtrado reales en la ayuda interna para poder
	# reutilizar exactamente la misma lógica en la función de visualización.
	return _pearson_correlation_filter(
		df=df,
		target_col=target_col,
		candidate_columns=candidate_columns,
		umbral_corr=umbral_corr,
		pvalue=pvalue,
	)


def plot_features_num_regression(
	df: pd.DataFrame,
	target_col: str = "",
	columns: list = [],
	umbral_corr: float = 0,
	pvalue: float = None,
) -> list:
	"""Representa pairplots de las variables numéricas seleccionadas y las devuelve.

	La fase de selección sigue exactamente la misma lógica de correlación que
	`get_features_num_regression`. La única diferencia es que esta función puede
	limitar el conjunto de candidatas mediante el argumento opcional `columns` y,
	además, representa el resultado con pairplots.

	Cuando la lista final de variables contiene más de cinco elementos, la
	función la divide en bloques de como máximo cuatro variables seleccionadas en
	cada iteración y siempre incluye `target_col` en cada gráfico. Así se
	mantiene cada visualización legible sin perder cobertura del conjunto total de
	columnas seleccionadas.

	Parámetros
	----------
	df:
		DataFrame de entrada.
	target_col:
		Columna objetivo numérica con la que comparar.
	columns:
		Lista opcional de columnas candidatas. Cuando está vacía, la función usa
		todas las columnas numéricas de `df` salvo la objetivo.
	umbral_corr:
		Umbral de correlación absoluta.
	pvalue:
		Umbral opcional de p-valor.

	Devuelve
	--------
	list
		Lista con los nombres de las variables que se han seleccionado y
		representado.
		Devuelve `None` cuando falla la validación.
	"""

	if not _validate_regression_inputs(df, target_col, umbral_corr, pvalue):
		return None

	# Si el usuario proporciona una lista explícita de candidatas, la respetamos,
	# pero seguimos quedándonos solo con columnas numéricas distintas del
	# objetivo.
	if columns:
		candidate_columns = [
			column
			for column in columns
			if column in df.columns
			and column != target_col
			and pd.api.types.is_numeric_dtype(df[column])
		]
	else:
		# En caso contrario, usamos la tipificación compartida para descubrir
		# candidatas automáticamente.
		tip = tipifica_variables(df, umbral_categoria=3, umbral_continua=50)
		if tip is None:
			candidate_columns = []
		else:
			candidate_columns = [
				col
				for col in tip.loc[tip["tipo_sugerido"].str.startswith("Numérica"), "nombre_variable"].tolist()
				if col in df.columns and col != target_col and pd.api.types.is_numeric_dtype(df[col])
			]

	# Reutilizamos exactamente las mismas reglas de filtrado que la función sin
	# visualización.
	selected_columns = _pearson_correlation_filter(
		df=df,
		target_col=target_col,
		candidate_columns=candidate_columns,
		umbral_corr=umbral_corr,
		pvalue=pvalue,
	)

	if not selected_columns:
		return []

	# Construimos grupos manejables para que conjuntos grandes de variables no
	# terminen en un único pairplot ilegible.
	groups = [selected_columns[index : index + 4] for index in range(0, len(selected_columns), 4)]

	for group in groups:
		# Pairplot necesita que la columna objetivo esté presente en cada
		# subconjunto representado.
		sns.pairplot(df[[target_col] + group].dropna())
		# Mostramos la figura explícitamente para que el efecto visual sea visible
		# en uso interactivo y en cuadernos.
		plt.show()

	return selected_columns
