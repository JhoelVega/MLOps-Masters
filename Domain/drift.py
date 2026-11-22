"""
Lógica de detección de drift (sin MLflow ni Prefect).

- Drift de features: Evidently (DataDriftPreset). KS para numéricas y chi² para binarias;
  una columna tiene drift si su p-valor < stattest_threshold.
- Drift de evento: chi² de bondad de ajuste entre la distribución de `price_range` del lote
  y la del entrenamiento, más la diferencia máxima de proporciones por clase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from scipy.stats import chisquare

from domain.models import BINARIAS, CLASES, NUMERICAS, OBJETIVO


class LoteSinEtiquetas(ValueError):
    """El lote no trae las etiquetas reales necesarias para el drift de evento."""


# ==========================
# Drift de features
# ==========================

@dataclass
class ResultadoDriftFeatures:
    p_valores: dict[str, float]
    columnas_con_drift: list[str]
    fraccion_con_drift: float
    umbral: float
    alerta: bool
    snapshot: object = field(default=None, repr=False)

    def resumen(self) -> dict:
        return {
            "n_columnas": len(self.p_valores),
            "columnas_con_drift": self.columnas_con_drift,
            "n_columnas_con_drift": len(self.columnas_con_drift),
            "fraccion_con_drift": round(self.fraccion_con_drift, 4),
            "umbral": self.umbral,
            "alerta": self.alerta,
            "p_valores": {k: float(f"{v:.6g}") for k, v in self.p_valores.items()},
        }


def _definicion(columnas: list[str]) -> DataDefinition:
    return DataDefinition(
        numerical_columns=[c for c in columnas if c in NUMERICAS],
        categorical_columns=[c for c in columnas if c in BINARIAS],
    )


def construir_reporte(columnas: list[str], stattest_threshold: float = 0.05) -> Report:
    return Report([
        DataDriftPreset(
            columns=columnas,
            num_method="ks",
            cat_method="chisquare",
            threshold=stattest_threshold,
        )
    ])


def extraer_p_valores(snapshot_dict: dict) -> dict[str, float]:
    """Lee el p-valor de cada `ValueDrift(column=...)` del snapshot de Evidently."""
    p_valores = {}
    for metrica in snapshot_dict.get("metrics", []):
        m = re.match(r"ValueDrift\(column=([^,)]+)", metrica.get("metric_id", ""))
        if m:
            p_valores[m.group(1)] = float(metrica["value"])
    return p_valores


def evaluar_p_valores(p_valores: dict[str, float], stattest_threshold: float, umbral_fraccion: float):
    """Devuelve (columnas_con_drift, fraccion, alerta)."""
    if not p_valores:
        raise ValueError("No hay p-valores para evaluar")
    con_drift = sorted(c for c, p in p_valores.items() if p < stattest_threshold)
    fraccion = len(con_drift) / len(p_valores)
    return con_drift, fraccion, debe_alertar(fraccion, umbral_fraccion)


def debe_alertar(fraccion: float, umbral: float) -> bool:
    return fraccion > umbral


def calcular_drift_features(
    referencia: pd.DataFrame,
    lote: pd.DataFrame,
    columnas: list[str],
    stattest_threshold: float = 0.05,
    umbral_fraccion: float = 0.30,
) -> ResultadoDriftFeatures:
    faltantes = [c for c in columnas if c not in lote.columns]
    if faltantes:
        raise ValueError(f"El lote no tiene las columnas: {faltantes}")

    definicion = _definicion(columnas)
    ref = Dataset.from_pandas(referencia[columnas], data_definition=definicion)
    cur = Dataset.from_pandas(lote[columnas], data_definition=definicion)
    snapshot = construir_reporte(columnas, stattest_threshold).run(cur, ref)

    p_valores = extraer_p_valores(snapshot.dict())
    con_drift, fraccion, alerta = evaluar_p_valores(p_valores, stattest_threshold, umbral_fraccion)
    return ResultadoDriftFeatures(p_valores, con_drift, fraccion, umbral_fraccion, alerta, snapshot)


# ==========================
# Drift de evento (target)
# ==========================

def validar_etiquetas(lote: pd.DataFrame) -> pd.Series:
    """Devuelve las etiquetas reales del lote o lanza LoteSinEtiquetas con un mensaje claro."""
    if OBJETIVO not in lote.columns:
        raise LoteSinEtiquetas(
            f"El lote no tiene la columna '{OBJETIVO}'. El drift de evento necesita las etiquetas "
            "reales; espera a que lleguen (pueden llegar con retraso) y vuelve a ejecutar."
        )
    y = lote[OBJETIVO]
    if y.isna().any():
        raise LoteSinEtiquetas(
            f"El lote tiene {int(y.isna().sum())} de {len(y)} filas sin etiqueta real en '{OBJETIVO}'. "
            "Completa las etiquetas antes de calcular el drift de evento."
        )
    invalidas = set(y.unique()) - set(CLASES)
    if invalidas:
        raise ValueError(f"Etiquetas fuera de {CLASES}: {sorted(invalidas)}")
    return y.astype(int)


def proporciones(y: pd.Series) -> pd.Series:
    return y.value_counts(normalize=True).reindex(CLASES, fill_value=0.0)


def calcular_drift_evento(y_referencia: pd.Series, lote: pd.DataFrame, alpha: float = 0.05) -> dict:
    y_lote = validar_etiquetas(lote)
    p_ref = proporciones(y_referencia)
    p_lote = proporciones(y_lote)

    observados = y_lote.value_counts().reindex(CLASES, fill_value=0).to_numpy()
    esperados = p_ref.to_numpy() * len(y_lote)
    estadistico, p_valor = chisquare(observados, esperados)

    diferencia = (p_lote - p_ref).abs()
    return {
        "n_lote": int(len(y_lote)),
        "proporcion_referencia": {int(k): round(float(v), 4) for k, v in p_ref.items()},
        "proporcion_lote": {int(k): round(float(v), 4) for k, v in p_lote.items()},
        "diferencia_max": round(float(diferencia.max()), 4),
        "chi2": round(float(estadistico), 4),
        "p_valor": float(f"{p_valor:.6g}"),
        "alpha": alpha,
        "drift": bool(p_valor < alpha),
        "alerta": bool(p_valor < alpha),
    }


def psi(referencia: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index entre dos vectores de proporciones."""
    r = np.clip(np.asarray(referencia, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - r) * np.log(a / r)))
