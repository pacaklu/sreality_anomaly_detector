"""Tests script for predicition method of LGBMModelInferor."""
from sreality_anomaly_detector.lgbm_inferor import LGBMModelInferor

FLAT_ID = 4065768524

config = {"model_path": r"C:\Users\pacak\PycharmProjects\sreality_detector\model\lgbm_model.pickle"}

if __name__ == "__main__":
    model = LGBMModelInferor(config)
    result = model.predict(FLAT_ID)
    print(result)
