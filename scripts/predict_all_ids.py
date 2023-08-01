import json
import pandas as pd
import requests

from sreality_anomaly_detector.configs import prediction_config
from scripts.email_sender import send_mail

def reconstruct_url_from_id(flat_id):
    """Reconstruct url from if of apartment to be easily visualise."""
    url = f"https://www.sreality.cz/api/cs/v2/estates/{flat_id}"
    obtained_json = requests.get(url=url)
    obtained_json = obtained_json.json()
    flat_locality = obtained_json["seo"]["locality"]
    if "2+1" in obtained_json["name"]["value"]:
        proportion = "2+1"
    else:
        proportion = "2+kk"

    url_obtained = f"https://www.sreality.cz/detail/prodej/byt/{proportion}/{flat_locality}/{flat_id}"
    return url_obtained


if __name__ == "__main__":
    data = pd.read_csv(prediction_config["data_path"])
    flat_ids_to_test = data["ID"].tolist()
    predictions = []
    urls = []

    for flat_id in flat_ids_to_test:
        API_ENDPOINT = prediction_config["api_url"] + str(flat_id)
        r = requests.post(url=API_ENDPOINT)
        predictions.append(r.json()['prediction_minus_actual_price'])
        urls.append(reconstruct_url_from_id(flat_id))

    final_data = pd.DataFrame([flat_ids_to_test, predictions, urls], columns=['flat_id', 'prediction_minus_actual', 'url'])
    final_data = final_data.sort_values(by = 'prediction_minus_actual', ascending = False).head(15)
    final_data.to_csv(prediction_config["data_path"], header = True, index = False)
    send_mail(prediction_config["data_path"])
