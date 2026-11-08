from prefect import flow, task

import mlflow
import mlflow.sklearn

import pandas as pd
import yaml

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from domain.Models import (
    construir_modelo,
    resolver_features,
    calcular_metricas,
    OBJETIVO,
)

MODEL_URI = "configs/config.yaml"
FEATURES_URI = "configs/features.yaml"


@task
def cargar_config():

    with open(MODEL_URI, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(FEATURES_URI, "r", encoding="utf-8") as f:
        features = yaml.safe_load(f)

    return config, features


@task
def cargar_datos():
    return pd.read_csv("data/mobile_price.csv")


@task
def entrenar(df, config, features_cfg):

    seleccion = features_cfg["seleccion"]

    if seleccion == "ALL":
        columnas = "ALL"
    elif seleccion == "CUSTOM":
        columnas = features_cfg["custom"]
    else:
        columnas = "ALL"

    numericas, categoricas = resolver_features(columnas)

    X = df[numericas + categoricas]
    y = df[OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    tipo = config["model"]["activo"]

    parametros = config["model"]["modelos"][tipo]

    modelo = construir_modelo(
        tipo,
        parametros,
        numericas,
        categoricas,
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)

    metricas = calcular_metricas(
        y_test,
        y_pred,
        y_prob,
    )

    return (
        modelo,
        metricas,
        tipo,
        len(X.columns),
    )


@task
def registrar_modelo(
    modelo,
    metricas,
    tipo_modelo,
    n_features,
    config,
):

    model_name = config["model"]["nombre"]

    mlflow.set_tag(
        "fase",
        "entrenamiento",
    )

    mlflow.log_param(
        "modelo",
        tipo_modelo,
    )

    mlflow.log_param(
        "n_features",
        n_features,
    )

    for nombre, valor in metricas.items():
        mlflow.log_metric(
            nombre,
            valor,
        )

    model_info = mlflow.sklearn.log_model(
        sk_model=modelo,
        artifact_path="model",
        registered_model_name=model_name,
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )

    client = MlflowClient()

    run_id = mlflow.active_run().info.run_id
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    run_versions = [
        version
        for version in versions
        if version.run_id == run_id
    ]
    if not run_versions:
        raise RuntimeError(
            f"MLflow no registró una versión para el run {run_id}"
        )

    nueva_version = max(
        run_versions,
        key=lambda version: int(version.version),
    ).version

    minimo_auc = config["evaluation"]["min_auc"]

    try:

        champion = client.get_model_version_by_alias(
            model_name,
            "champion",
        )

        champion_run = client.get_run(
            champion.run_id
        )

        auc_actual = float(
            champion_run.data.metrics["auc"]
        )

    except Exception:
        auc_actual = 0

    nuevo_auc = metricas["auc"]

    print(f"AUC Champion: {auc_actual}")
    print(f"AUC Nuevo: {nuevo_auc}")

    if (
        nuevo_auc >= minimo_auc
        and nuevo_auc > auc_actual
    ):
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=nueva_version,
        )

        print(
            f"Champion actualizado -> versión {nueva_version}"
        )


@flow(name="training-flow")
def training_flow():

    config, features = cargar_config()

    df = cargar_datos()

    modelo, metricas, tipo_modelo, n_features = entrenar(
        df,
        config,
        features,
    )

    with mlflow.start_run():

        registrar_modelo(
            modelo,
            metricas,
            tipo_modelo,
            n_features,
            config,
        )

    print(metricas)


if __name__ == "__main__":
    training_flow()
