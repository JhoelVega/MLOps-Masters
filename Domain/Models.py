"""
Lógica de negocio del modelo de clasificación de precios de teléfonos móviles.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================
# Variables del problema
# ==========================

NUMERICAS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]

CATEGORICAS = []

VARIABLES = NUMERICAS + CATEGORICAS

OBJETIVO = "price_range"

# ==========================
# Selección de features
# ==========================

def resolver_features(columnas):
    """
    Convierte la configuración
    ('ALL' o lista personalizada)
    en variables numéricas y categóricas.
    """

    if columnas == "ALL":
        elegidas = VARIABLES

    else:
        desconocidas = [
            v for v in columnas
            if v not in VARIABLES
        ]

        if desconocidas:
            raise ValueError(
                f"Variables desconocidas: {desconocidas}"
            )

        elegidas = columnas

    if not elegidas:
        raise ValueError(
            "La selección de variables quedó vacía"
        )

    numericas = [
        v for v in NUMERICAS
        if v in elegidas
    ]

    categoricas = [
        v for v in CATEGORICAS
        if v in elegidas
    ]

    return numericas, categoricas


# ==========================
# Construcción del modelo
# ==========================

def construir_modelo(
    tipo,
    parametros,
    numericas,
    categoricas,
):
    """
    Construye el pipeline completo: preprocesamiento + modelo.
    """

    if tipo == "random_forest":

        estimador = RandomForestClassifier(
            **parametros
        )

    elif tipo == "xgboost":

        estimador = XGBClassifier(
            **parametros,
            eval_metric="mlogloss",
            objective="multi:softprob"
        )

    else:
        raise ValueError(
            f"Modelo '{tipo}' no soportado"
        )

    transformadores = []

    if numericas:
        transformadores.append(
            (
                "num",
                StandardScaler(),
                numericas,
            )
        )

    preprocesado = ColumnTransformer(
        transformadores
    )

    return Pipeline(
        [
            ("preprocesado", preprocesado),
            ("modelo", estimador),
        ]
    )

# ==========================
# Métricas
# ==========================

def calcular_metricas(
    y_true,
    y_pred,
    y_prob,
):
    """
    Métricas para clasificación multiclase.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
            )
        ),
        "auc": float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
            )
        ),
    }
