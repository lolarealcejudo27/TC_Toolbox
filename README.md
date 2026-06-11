# TC_Toolbox
Team Challenge porpuesto pot The Bridge para la creación de un toolbox.
# TC_Toolbox

**`toolbox_ml`** es un paquete de Python construido desde cero que automatiza el Análisis Exploratorio de Datos (EDA) y la selección de predictores para modelos de regresión. A diferencia de un script básico de IA, este proyecto sigue los estándares de un equipo de Data Science profesional: está completamente modularizado, incluye tests automatizados con `pytest`, cuenta con gestión de dependencias integrada (`setup.py`) y genera paneles visuales avanzados listos para producción.

---

## Estructura del Proyecto

El proyecto sigue la siguiente organización de directorios:

**`notebooks/`**
* `demo.ipynb` → Jupyter Notebook con el flujo de demostración de las funciones.

**`tests/`**
* `test_core.py` → Tests unitarios para validar el correcto funcionamiento de las funciones.

**`toolbox_ml/`**
* **`eda/`**
  * `core.py` → Módulo principal con la lógica algorítmica y estadística.
  * `__init__.py` → Inicializador para la importación del paquete.

**Archivos base**
* `.gitignore` → Configuración para excluir archivos temporales y entornos virtuales.
* `README.md` → Documentación principal del repositorio.
* `requirements.txt` → Dependencias requeridas por el paquete.
* `setup.py` → Archivo de configuración para instalar el paquete de forma local.

---

## Instrucciones de Instalación

Sigue estos pasos para clonar el repositorio, configurar el entorno virtual e instalar el paquete en modo desarrollo:

```bash
#1. Clonar el repositorio
git clone [https://github.com/lolarealcejudo27/TC_Toolbox.git](https://github.com/lolarealcejudo27/TC_Toolbox.git)

#2. Acceder a la carpeta del proyecto
cd TC_Toolbox

#3. Crear el entorno virtual (venv)
python -m venv venv

#4. Activar el entorno virtual
#En Mac/Linux:
source venv/bin/activate
#En Windows (PowerShell):
#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
#. .\venv\Scripts\Activate.ps1

#5. Instalar las dependencias del proyecto
pip install -r requirements.txt

#6. Instalar el paquete local en modo editable (desarrollo)
pip install -e .
```

---

## Ejemplos de Uso (Código Ejecutable)

A continuación se detalla cómo importar y ejecutar cada una de las funciones públicas incluidas en el módulo `toolbox_ml.eda`. Para estos ejemplos, asumimos que se utiliza el dataset del Titanic (`df_titanic`) y que nuestra variable objetivo es la tarifa del billete (`target = "fare"`).

### 1.Análisis Exploratorio Básico
```python
from toolbox_ml.eda.core import tipifica_variables, describe_df

#Tipificación automática de variables según su cardinalidad y tipo de datos
df_tipificado = tipifica_variables(df_titanic, umbral_categorica=10, umbral_continua=30)
print(df_tipificado)

#Resumen estructurado del estado del DataFrame (nulos, tipos, cardinalidad, etc.)
resumen_estructura = describe_df(df_titanic)
print(resumen_estructura)
```

### 2.Relación Numérica vs Numérica (Regresión)
```python
from toolbox_ml.eda.core import get_features_num_regression, plot_features_num_regression

#Filtramos predictores numéricos basados en correlación de Pearson y significancia estadística
num_features_filtradas = get_features_num_regression(df=df_titanic, target_col="fare", umbral_corr=0.10, pvalue=0.05)
print(f"Variables numéricas significativas: {num_features_filtradas}")

#Generamos la matriz de gráficos de dispersión (pairplot) para las variables seleccionadas
plot_features_num_regression(df=df_titanic, target_col="fare", umbral_corr=0.10, pvalue=0.05)
```

### 3.Relación Categórica vs Numérica (Regresión)
```python
from toolbox_ml.eda.core import get_features_cat_regression, plot_features_cat_regression

#Filtramos predictores cualitativos aplicando Mann-Whitney U o ANOVA de un factor
cat_features_filtradas = get_features_cat_regression(df=df_titanic, target_col="fare", pvalue=0.05)
print(f"Variables categóricas significativas: {cat_features_filtradas}")

#Generamos un panel con histogramas solapados y transparentes por grupos
plot_features_cat_regression(df=df_titanic, target_col="fare", columns=cat_features_filtradas, pvalue=0.05, with_individual_plot=False)
```

---

## Cómo Ejecutar los Tests

El proyecto cuenta con una batería de pruebas automatizadas con `pytest` para garantizar la robustez de las validaciones de datos internas y el control de errores.

Para ejecutar los tests de forma detallada, asegúrate de tener el entorno activado y lanza:

```bash
pytest tests/ -v
```

---

## Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* SciPy (Pruebas estadísticas)
* Matplotlib & Seaborn (Visualización avanzada)
* Pytest (Tests unitarios)

---

## Descripción del Equipo y Reparto de Tareas

Este proyecto ha sido desarrollado de forma colaborativa siguiendo metodologías ágiles (Scrum) y buenas prácticas de ingeniería de software.

| Integrante | Rol | Responsabilidad |
| :--- | :--- | :--- |
| **Lola** | Scrum Master | Gestión y configuración inicial del repo, diseño y mantenimiento de `setup.py`, inicializadores `__init__.py`, resolución de rutas de importación, integración final y maquetación técnica de `demo.ipynb`. |
| **Miguel** | Desarrollador 1 | Diseño y desarrollo del bloque EDA básico. Lógica de las funciones `describe_df` y `tipifica_variables`, junto con la codificación de sus pruebas unitarias automatizadas. |
| **Emilio** | Desarrollador 2 | Bloque analítico Numérica vs Numérica. Desarrollo de `get_features_num_regression` (Pearson/p-valores) y la herramienta visual `plot_features_num_regression` (pairplot) con sus tests unitarios. |
| **Sandra** | Desarrollador 3 | Bloque analítico Categórica vs Numérica. Lógica de `get_features_cat_regression` (Mann-Whitney U / ANOVA), diseño gráfico de `plot_features_cat_regression` (subplots), función bonus y sus tests. |
