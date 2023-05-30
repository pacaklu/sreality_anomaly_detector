import pandas as pd
import requests

from sreality_anomaly_detector.configs import prediction_config

if __name__ == "__main__":

    data = pd.read_csv(prediction_config["data_path"])
    flat_ids_to_test = data["id"].tolist()

    for flat_id in flat_ids_to_test:
        API_ENDPOINT = prediction_config["api_url"] + flat_id
        r = requests.post(url=API_ENDPOINT)
