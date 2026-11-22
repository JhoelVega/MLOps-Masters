"""Lógica de la inferencia por lotes, independiente de MLflow/Prefect (el modelo se inyecta)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from domain.drift_emulado import emular_drift
from domain.models import VARIABLES
from domain.validacion import validar_datos

COLUMNA_PREDICCION = "prediccion"


def columnas_del_modelo(modelo) -> list[str]:
    """Columnas de entrada según la firma del modelo MLflow; si no tiene firma, todas las VARIABLES."""
    try:
        schema = modelo.metadata.get_input_schema()
        if schema is not None:
            return list(schema.input_names())
    except AttributeError:
        pass
    return list(VARIABLES)


def puntuar_lote(modelo, lote: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Devuelve una copia del lote con la columna de predicción."""
    faltantes = [c for c in features if c not in lote.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas para predecir: {faltantes}")
    salida = lote.copy()
    salida[COLUMNA_PREDICCION] = [int(p) for p in modelo.predict(lote[features])]
    return salida


def nombre_lote(nivel: float, momento: datetime | None = None) -> str:
    momento = momento or datetime.now()
    return f"lote_{momento:%Y%m%d_%H%M%S_%f}_drift{int(round(nivel * 100)):03d}"


def guardar_lote(lote: pd.DataFrame, info: dict, directorio: str | Path, nombre: str) -> Path:
    """Guarda el CSV puntuado y un JSON con los metadatos de la emulación."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{nombre}.csv"
    lote.to_csv(ruta, index=False)
    (directorio / f"{nombre}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def ejecutar_inferencia(modelo, pool_test: pd.DataFrame, cfg_inferencia: dict, directorio_salida, momento=None) -> tuple[Path, dict]:
    """Ingesta → drift emulado → validación de esquema → predicción → copia del lote."""
    nivel = cfg_inferencia.get("drift_level", 0.0)
    lote, info = emular_drift(
        pool_test,
        nivel=nivel,
        batch_size=int(cfg_inferencia.get("batch_size", 200)),
        random_state=int(cfg_inferencia.get("random_state", 42)),
    )
    # El drift puede sacar valores de rango (es lo que se quiere detectar): solo se valida esquema y nulos
    validar_datos(lote, requiere_objetivo=True, verificar_rangos=False)

    puntuado = puntuar_lote(modelo, lote, columnas_del_modelo(modelo))
    nombre = nombre_lote(info["drift_level"], momento)
    info = {**info, "lote": nombre, "creado": (momento or datetime.now()).isoformat(timespec="seconds")}
    ruta = guardar_lote(puntuado, info, directorio_salida, nombre)
    return ruta, info
