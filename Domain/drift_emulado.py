"""
Emulación controlada de drift sobre lotes tomados de la partición de prueba.

Un único parámetro `nivel` en [0, 1] controla dos efectos:

1. Drift de evento (prior shift del target): la proporción de clases del lote se mezcla
   entre la del pool (≈ 25 % por clase) y una distribución sesgada a gama alta:
       p(nivel) = (1 - nivel) · p_pool + nivel · P_DRIFT,   P_DRIFT = [0.10, 0.10, 0.20, 0.60]
   Las filas se muestrean por clase desde el pool, así que las variables
   correlacionadas con el precio (sobre todo `ram`) también se mueven.

2. Drift de features (covariate shift): se desplazan las primeras
   round(nivel · 12) variables de FEATURES_CON_DRIFT sumando
       nivel · MAGNITUD · σ_pool(col)
   (σ calculada sobre el pool, valores redondeados al tipo original y sin negativos).

Con nivel 0.0 el lote es una muestra estratificada normal, sin modificaciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.models import CLASES, OBJETIVO

FEATURES_CON_DRIFT = [
    "ram",
    "battery_power",
    "px_height",
    "px_width",
    "int_memory",
    "mobile_wt",
    "talk_time",
    "clock_speed",
    "fc",
    "pc",
    "sc_h",
    "sc_w",
]
P_DRIFT = np.array([0.10, 0.10, 0.20, 0.60])
MAGNITUD = 1.5


def validar_nivel(nivel: float) -> float:
    nivel = float(nivel)
    if not 0.0 <= nivel <= 1.0:
        raise ValueError(f"drift_level debe estar entre 0.0 y 1.0 (recibido {nivel})")
    return nivel


def distribucion_objetivo(p_base: np.ndarray, nivel: float) -> np.ndarray:
    """Mezcla lineal entre la distribución base y P_DRIFT."""
    p = (1 - nivel) * np.asarray(p_base, dtype=float) + nivel * P_DRIFT
    return p / p.sum()


def features_afectadas(nivel: float) -> list[str]:
    return FEATURES_CON_DRIFT[: int(round(nivel * len(FEATURES_CON_DRIFT)))]


def emular_drift(pool: pd.DataFrame, nivel: float, batch_size: int, random_state: int = 42):
    """
    Construye un lote de `batch_size` filas desde `pool` (partición de prueba, con etiquetas)
    y le aplica el drift emulado. Devuelve (lote, info).
    """
    nivel = validar_nivel(nivel)
    if OBJETIVO not in pool.columns:
        raise ValueError(f"El pool debe incluir la columna objetivo '{OBJETIVO}'")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo")

    rng = np.random.default_rng(random_state)

    # 1) Prior shift del objetivo
    p_base = pool[OBJETIVO].value_counts(normalize=True).reindex(CLASES, fill_value=0).to_numpy()
    p_lote = distribucion_objetivo(p_base, nivel)
    conteos = rng.multinomial(batch_size, p_lote)

    partes = []
    for clase, n in zip(CLASES, conteos):
        if n == 0:
            continue
        candidatas = pool[pool[OBJETIVO] == clase]
        if candidatas.empty:
            continue
        idx = rng.choice(candidatas.index.to_numpy(), size=n, replace=n > len(candidatas))
        partes.append(pool.loc[idx])
    lote = pd.concat(partes).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 2) Covariate shift
    afectadas = features_afectadas(nivel)
    for col in afectadas:
        desplazamiento = nivel * MAGNITUD * pool[col].std()
        nuevo = (lote[col] + desplazamiento).clip(lower=0)
        if pd.api.types.is_integer_dtype(pool[col]):
            nuevo = nuevo.round().astype(pool[col].dtype)
        else:
            nuevo = nuevo.round(2)
        lote[col] = nuevo

    info = {
        "drift_level": nivel,
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "features_con_drift": afectadas,
        "proporcion_objetivo_esperada": {int(c): round(float(p), 4) for c, p in zip(CLASES, p_lote)},
    }
    return lote, info
