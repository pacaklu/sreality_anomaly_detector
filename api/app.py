"""API for LGBMINFEROR class."""
from fastapi import FastAPI
from sreality_anomaly_detector.configs import inference_model_config
from sreality_anomaly_detector.lgbm_inferor import LGBMModelInferor

model = LGBMModelInferor(inference_model_config)

app = FastAPI()


@app.get("/")
def root():
    """Intro message."""
    return {"Welcome to default endpoint for flat price prediction."}


@app.post("/predict")
def predict(input_data):
    """Predict method of the model class."""
    print(input_data)
    return model.predict(input_data)
