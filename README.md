# MLOps-Masters
Repositorio actividades MLOps

Proyecto académico para la actividad de Fundamentos, Versionamiento y Tracking.

# Paso 1. Estructura y entorno
Creamos un repositorio de GitHub, lo hizo un integrante y agregó a los demás como colaboradores
# Requisitos
Python 3.12 o superior
Poetry
# Clonar el repositorio en el computador
git clone https://github.com/JhoelVega/MLOps-Masters.git
Creamos la estructura y el entorno
# Instalar dependencias
Dependencias principales
Prefect
MLflow
Scikit-learn
Pandas
NumPy
PyYAML
# Clonar y preparar el entorno (los demás integrantes del equipo):

# Paso 2. Caso de uso
Se utilizará un dataset descargado de Kaggle para realizar la predicción del nivel de obesidad de una persona a partir de hábitos alimenticios, actividad física y características demográficas.

Dataset: Obesity Levels Dataset (Kaggle). https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels

Variable objetivo: NObeyesdad.

Tipo de problema: Clasificación multiclase, debido a que el modelo debe predecir uno de varios niveles de obesidad definidos en la variable objetivo NObeyesdad.

Objetivo del modelo: Estimar el nivel de obesidad de una persona para apoyar la identificación temprana de factores de riesgo asociados a sobrepeso y obesidad.

# Paso 3. Levantar los servicios y comprobar Prefect