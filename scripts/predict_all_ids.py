import json
import logging
import pandas as pd
import requests

from sreality_anomaly_detector.configs import prediction_config

if __name__ == "__main__":
    data = pd.read_csv(prediction_config["data_path"])
    flat_ids_to_test = data["ID"].tolist()

    for flat_id in flat_ids_to_test:
        API_ENDPOINT = prediction_config["api_url"] + str(flat_id)
        r = requests.post(url=API_ENDPOINT)
        logging.INFO(f'prediction for {flat_id} is {r.json()}')
