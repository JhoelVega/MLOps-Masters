"""
Lógica de negocio del modelo de clasificación de rango de precios de teléfonos móviles.

Todo lo de este módulo es puro (sin MLflow ni Prefect) para poder probarlo en aislamiento.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ==========================
# Variables del problema
# ==========================

# Variables continuas / discretas con muchos valores
NUMERICAS = [
    "battery_power",
    "clock_speed",
    "fc",
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
]

# Variables binarias 0/1 (se tratan como categóricas en el análisis de drift)
BINARIAS = [
    "blue",
    "dual_sim",
    "four_g",
    "three_g",
    "touch_screen",
    "wifi",
]

CATEGORICAS: list[str] = []

VARIABLES = NUMERICAS + BINARIAS + CATEGORICAS

OBJETIVO = "price_range"
CLASES = [0, 1, 2, 3]


# ==========================
# Selección de features
# ==========================

def columnas_desde_config(features_cfg: dict) -> str | list[str]:
    """Traduce configs/features.yaml a 'ALL' o a la lista personalizada."""
    seleccion = str(features_cfg.get("seleccion", "ALL")).upper()
    if seleccion == "ALL":
        return "ALL"
    if seleccion == "CUSTOM":
        return list(features_cfg.get("custom") or [])
    raise ValueError(f"seleccion debe ser ALL o CUSTOM, no '{seleccion}'")


def resolver_features(columnas) -> list[str]:
    """Devuelve la lista ordenada de variables a usar ('ALL' o lista personalizada)."""
    if columnas == "ALL":
        elegidas = list(VARIABLES)
    else:
        desconocidas = [v for v in columnas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"Variables desconocidas: {desconocidas}")
        elegidas = [v for v in VARIABLES if v in columnas]

    if not elegidas:
        raise ValueError("La selección de variables quedó vacía")
    return elegidas


# ==========================
# Partición de datos
# ==========================

def separar_datos(df: pd.DataFrame, test_size: float, random_state: int):
    """Partición estratificada train/test. Devuelve dos DataFrames con el objetivo incluido."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[OBJETIVO],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ==========================
# Construcción del modelo
# ==========================

def construir_modelo(tipo: str, parametros: dict, features: list[str]) -> Pipeline:
    """
    Pipeline: selección de columnas + modelo.

    Los modelos de árboles no necesitan escalado, así que el ColumnTransformer
    solo garantiza que el modelo reciba exactamente las columnas elegidas y en orden.
    """
    if tipo == "random_forest":
        estimador = RandomForestClassifier(**parametros)
    elif tipo == "xgboost":
        estimador = XGBClassifier(
            **parametros,
            eval_metric="mlogloss",
            objective="multi:softprob",
        )
    else:
        raise ValueError(f"Modelo '{tipo}' no soportado")

    preprocesado = ColumnTransformer(
        [("features", "passthrough", features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def entrenar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame, tipo: str, parametros: dict, features: list[str]):
    """Entrena con train y evalúa con test. Devuelve (modelo, métricas)."""
    modelo = construir_modelo(tipo, parametros, features)
    modelo.fit(train[features], train[OBJETIVO])
    y_pred = modelo.predict(test[features])
    y_prob = modelo.predict_proba(test[features])
    return modelo, calcular_metricas(test[OBJETIVO], y_pred, y_prob)


# ==========================
# Métricas
# ==========================

def calcular_metricas(y_true, y_pred, y_prob) -> dict:
    """Métricas para clasificación multiclase."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "auc": float(roc_auc_score(y_true, y_prob, multi_class="ovr")),
    }


def es_nuevo_champion(nuevo_auc: float, auc_champion: float | None, minimo_auc: float) -> bool:
    """El modelo nuevo es champion si supera el mínimo y mejora al champion actual (si existe)."""
    if nuevo_auc < minimo_auc:
        return False
    return auc_champion is None or nuevo_auc > auc_champion
        "auc": float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
            )
        ),
    }
