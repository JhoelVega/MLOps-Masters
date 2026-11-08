from prefect import flow

import pandas as pd
import mlflow
from datetime import datetime


@flow(name="inference-flow")
def inference_flow():

    model = mlflow.pyfunc.load_model(
        "models:/mobile_price_classifier@champion"
    )

    datos = pd.read_csv(
        "data/mobile_price.csv"
    ).head(50)

    predicciones = model.predict(datos)

    datos["prediccion"] = predicciones

    fecha = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    salida = (
        f"predictions/predicciones_{fecha}.csv"
    )

    datos.to_csv(
        salida,
        index=False,
    )

    print(f"Archivo generado: {salida}")


if __name__ == "__main__":
    inference_flow()
