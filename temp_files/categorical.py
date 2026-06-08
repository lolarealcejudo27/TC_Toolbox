import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, f_oneway
import matplotlib.pyplot as plt


def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Identifica variables categóricas significativamente relacionadas
    con una variable target numérica mediante tests estadísticos.

    - 2 categorías → Mann-Whitney U
    - >2 categorías → ANOVA de un factor
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
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
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
    - 2 categorías  → Mann-Whitney U
    - >2 categorías → ANOVA de un factor

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
        columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    # -------------------------
    # TEST ESTADÍSTICO Y SELECCIÓN
    # -------------------------
    selected_features = []

    for col in columns:
        if col not in df.columns:
            continue

        groups = df[col].dropna().unique()
        if len(groups) < 2:
            continue

        if len(groups) == 2:
            group1 = df[df[col] == groups[0]][target_col].dropna()
            group2 = df[df[col] == groups[1]][target_col].dropna()
            if len(group1) == 0 or len(group2) == 0:
                continue
            _, p = mannwhitneyu(group1, group2, alternative="two-sided")
        else:
            samples = []
            for g in groups:
                s = df[df[col] == g][target_col].dropna()
                if len(s) > 0:
                    samples.append(s)
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
            for g in df[col].dropna().unique():
                data = df[df[col] == g][target_col].dropna()
                ax.hist(data, bins=15, alpha=0.5, label=str(g))
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
            for g in df[col].dropna().unique():
                data = df[df[col] == g][target_col].dropna()
                axes[i].hist(data, bins=15, alpha=0.5, label=str(g))
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].legend(title=col)

        # ocultar subplots vacíos si el número de variables es impar
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Variables categóricas significativas vs {target_col}",
            fontsize=13,
            fontweight="bold"
        )
        plt.tight_layout()

    return selected_features