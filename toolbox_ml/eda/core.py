import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, mannwhitneyu, f_oneway


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
	"""Filtra columnas candidatas usando correlación de Pearson y p-valor opcional."""
	selected_columns: list[str] = []

	for column in candidate_columns:
		pair = df[[column, target_col]].dropna()
		if len(pair) < 2:
			continue
		try:
			corr, p_val = pearsonr(pair[column], pair[target_col])
		except Exception:
			continue
		if pd.isna(corr) or abs(corr) < umbral_corr:
			continue
		if pvalue is not None and p_val >= pvalue:
			continue
		selected_columns.append(column)
	return selected_columns


def get_features_num_regression(
	df: pd.DataFrame,
	target_col: str,
	umbral_corr: float,
	pvalue: float = None,
	umbral_categoria: int = 4,
	umbral_continua: float = 10,
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
	umbral_categoria:
		Umbral de cardinalidad para `tipifica_variables`. Columnas con menos
		valores únicos se consideran categóricas.
	umbral_continua:
		Umbral de porcentaje de cardinalidad para `tipifica_variables`. Por
		encima de este valor una variable numérica se considera continua.

	Devuelve
	--------
	list
		Lista con los nombres de las columnas que cumplen las reglas de
		correlación.
		Devuelve `None` cuando falla la validación de entrada.
	"""

	if not _validate_regression_inputs(df, target_col, umbral_corr, pvalue):
		return None

	tipos = tipifica_variables(df, umbral_categoria, umbral_continua)
	candidate_columns = tipos.loc[
		tipos["tipo_sugerido"].isin(["Numérica Continua", "Numérica Discreta"])
		& (tipos["nombre_variable"] != target_col),
		"nombre_variable",
	].tolist()
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
	umbral_categoria: int = 4,
	umbral_continua: float = 10,
	with_individual_plot: bool = False,
) -> list:
	"""Representa pairplots de las variables numéricas seleccionadas junto a target_col.

	La fase de selección sigue exactamente la misma lógica de correlación que
	`get_features_num_regression`. La única diferencia es que esta función puede
	limitar el conjunto de candidatas mediante el argumento opcional `columns` y,
	además, representa el resultado como un pairplot de seaborn.

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
	umbral_categoria:
		Umbral de cardinalidad para `tipifica_variables`. Columnas con menos
		valores únicos se consideran categóricas.
	umbral_continua:
		Umbral de porcentaje de cardinalidad para `tipifica_variables`. Por
		encima de este valor una variable numérica se considera continua.
	with_individual_plot:
		Si False (por defecto), un único pairplot con todas las variables
		seleccionadas más target_col. Si True, un pairplot por cada variable
		emparejada individualmente con target_col.
	Devuelve
	--------
	list
		Lista con los nombres de las variables que se han seleccionado y
		representado.
		Devuelve `None` cuando falla la validación.
	"""

	if not _validate_regression_inputs(df, target_col, umbral_corr, pvalue):
		return None

	tipos = tipifica_variables(df, umbral_categoria, umbral_continua)
	numeric_cols = set(
		tipos.loc[
			tipos["tipo_sugerido"].isin(["Numérica Continua", "Numérica Discreta"])
			& (tipos["nombre_variable"] != target_col),
			"nombre_variable",
		].tolist()
	)

	if columns:
		candidate_columns = [c for c in columns if c in numeric_cols]
	else:
		candidate_columns = [c for c in tipos["nombre_variable"] if c in numeric_cols]

	selected_columns = _pearson_correlation_filter(
		df=df,
		target_col=target_col,
		candidate_columns=candidate_columns,
		umbral_corr=umbral_corr,
		pvalue=pvalue,
	)

	if not selected_columns:
		return []

	palette = sns.color_palette("husl", n_colors=len(selected_columns))
	scatter_kws = {"alpha": 0.5, "edgecolors": "none", "s": 20}
	diag_kws = {"fill": True, "alpha": 0.4}

	if with_individual_plot:
		for i, col in enumerate(selected_columns):
			data = df[[target_col, col]].dropna()
			g = sns.pairplot(
				data,
				diag_kind="kde",
				plot_kws={**scatter_kws, "color": palette[i]},
				diag_kws={**diag_kws, "color": palette[i]},
			)
			g.figure.suptitle(f"{col} vs {target_col}", y=1.02, fontsize=13, fontweight="bold")
			plt.tight_layout()
	else:
		plot_cols = [target_col] + selected_columns
		data = df[plot_cols].dropna()
		g = sns.pairplot(
			data,
			diag_kind="kde",
			plot_kws={**scatter_kws, "color": "#4C72B0"},
			diag_kws={**diag_kws, "color": "#D8A61B"},
		)
		for ax in g.axes.flatten():
			if ax is not None:
				ax.set_facecolor("#f9f9f9")
		g.figure.suptitle(
			f"Variables numéricas seleccionadas vs {target_col}",
			y=1.02,
			fontsize=13,
			fontweight="bold",
		)
		plt.tight_layout()

	return selected_columns


def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Identifica variables categóricas significativamente relacionadas
    con una variable target numérica mediante tests estadísticos.

    - 2 categorías: Mann-Whitney U
    - >2 categorías: ANOVA de un factor
    """

    # -------------------------
    # VALIDACIONES
    # -------------------------
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame")
        return None

    if target_col not in df.columns:
        print("Error: target_col no existe en el DataFrame")
        return None

    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: pvalue inválido")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérica")
        return None

    # -------------------------
    # SELECCIÓN DE VARIABLES CATEGÓRICAS
    # -------------------------
    cat_cols = df.select_dtypes(include=["str", "category", "bool"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    selected_features = []

    # -------------------------
    # TEST ESTADÍSTICO POR VARIABLE
    # -------------------------
    for col in cat_cols:

        # categorías únicas sin NaN
        groups = df[col].dropna().unique()

        # no se puede testear con menos de 2 grupos
        if len(groups) < 2:
            continue

        # -------------------------
        # CASO 1: 2 categorías → Mann-Whitney U
        # -------------------------
        if len(groups) == 2:

            group1 = df[df[col] == groups[0]][target_col].dropna()
            group2 = df[df[col] == groups[1]][target_col].dropna()

            if len(group1) == 0 or len(group2) == 0:
                continue

            _, p = mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

        # -------------------------
        # CASO 2: >2 categorías → ANOVA
        # -------------------------
        else:

            samples = []
            for g in groups:
                s = df[df[col] == g][target_col].dropna()
                if len(s) > 0:
                    samples.append(s)

            if len(samples) < 2:
                continue

            _, p = f_oneway(*samples)

        # -------------------------
        # DECISIÓN FINAL
        # -------------------------
        if p < pvalue:
            selected_features.append(col)

    return selected_features


def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list:
    """
    Para cada variable categórica de `columns` que supere el test estadístico
    correspondiente, pinta histogramas solapados de target_col agrupados por
    cada valor de esa variable. Devuelve la lista de columnas que han superado
    el test y se han representado.

    Si `columns` está vacío, usa automáticamente todas las columnas categóricas
    del DataFrame como candidatas.

    El test estadístico aplicado es:
    - 2 categorías: Mann-Whitney U
    - >2 categorías: ANOVA de un factor

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    target_col : str
        Nombre de la variable numérica a analizar.
    columns : list
        Lista de columnas categóricas candidatas. Si está vacía, se infieren
        automáticamente del DataFrame.
    pvalue : float
        Umbral de significancia estadística. Por defecto 0.05.
    with_individual_plot : bool
        Si False (por defecto), todas las variables se representan en una única
        figura con subplots. Si True, cada variable genera su propia figura.

    Retorna
    -------
    list
        Lista de columnas categóricas que superaron el test estadístico.
        Devuelve None si alguna validación de entrada falla.
    """

    # -------------------------
    # VALIDACIONES
    # -------------------------
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame")
        return None
    if target_col not in df.columns:
        print("Error: target_col no existe en el DataFrame")
        return None
    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: pvalue inválido")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérica")
        return None

    # -------------------------
    # SELECCIÓN DE COLUMNAS CANDIDATAS
    # -------------------------
    if not columns:
        columns = df.select_dtypes(include=["str", "category", "bool"]).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    # -------------------------
    # TEST ESTADÍSTICO Y SELECCIÓN
    # -------------------------
    selected_features = []

    for col in columns:
        if col not in df.columns:
            continue
        categories = df[col].dropna().unique()
        if len(categories) < 2:
            continue
        if len(categories) == 2:
            group1 = df[df[col] == categories[0]][target_col].dropna()
            group2 = df[df[col] == categories[1]][target_col].dropna()
            if len(group1) == 0 or len(group2) == 0:
                continue
            _, p = mannwhitneyu(group1, group2, alternative="two-sided")
        else:
            samples = []
            for category in categories:
                sample = df[df[col] == category][target_col].dropna()
                if len(sample) > 0:
                    samples.append(sample)
            if len(samples) < 2:
                continue
            _, p = f_oneway(*samples)
        if p < pvalue:
            selected_features.append(col)

    # -------------------------
    # VISUALIZACIÓN
    # -------------------------
    if not selected_features:
        print("Ninguna variable superó el test estadístico.")
        return selected_features

    if with_individual_plot:
        # una figura independiente por cada variable significativa
        for col in selected_features:
            fig, ax = plt.subplots(figsize=(8, 4))
            for category in df[col].dropna().unique():
                data = df[df[col] == category][target_col].dropna()
                ax.hist(data, bins=15, alpha=0.5, label=str(category))
            ax.set_title(col, fontsize=11)
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frecuencia")
            ax.legend(title=col)
            fig.suptitle(f"{col} vs {target_col}", fontsize=13, fontweight="bold")
            plt.tight_layout()
    else:
        # una única figura con subplots para todas las variables significativas
        n = len(selected_features)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(selected_features):
            for category in df[col].dropna().unique():
                data = df[df[col] == category][target_col].dropna()
                axes[i].hist(data, bins=15, alpha=0.5, label=str(category))
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].legend(title=col)

        # ocultar subplots vacíos si el número de variables es impar
        for empty_ax_idx in range(i + 1, len(axes)):
            axes[empty_ax_idx].set_visible(False)
        fig.suptitle(
            f"Variables categóricas significativas vs {target_col}",
            fontsize=13,
            fontweight="bold"
        )
        plt.tight_layout()
    return selected_features