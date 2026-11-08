from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd


app = FastAPI(
    title="Mobile Price Prediction API"
)

MODEL_NAME = (
    "models:/mobile_price_classifier@champion"
)

model = mlflow.pyfunc.load_model(
    MODEL_NAME
)


class MobileFeatures(BaseModel):

    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int


@app.get("/")
def home():

    return {
        "message": "Mobile Price Prediction API"
    }


@app.post("/predict")
def predict(data: MobileFeatures):

    df = pd.DataFrame([data.model_dump()])

    pred = model.predict(df)

    return {
        "price_range": int(pred[0])
    }
