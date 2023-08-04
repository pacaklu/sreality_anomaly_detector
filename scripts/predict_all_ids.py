"""Predict results for all scraped flats."""
import logging

import pandas as pd
import requests
from tqdm import tqdm
from sreality_anomaly_detector.configs import prediction_config

from scripts.email_sender import send_mail

FLATS_TO_SEND = 15

def reconstruct_url_from_id(flat_id: int):
    """Reconstruct url from if of apartment to be easily visualise."""
    url = f"https://www.sreality.cz/api/cs/v2/estates/{flat_id}"
    obtained_json = requests.get(url=url)
    obtained_json = obtained_json.json()
    flat_locality = obtained_json["seo"]["locality"]
    if "2+1" in obtained_json["name"]["value"]:
        proportion = "2+1"
    else:
        proportion = "2+kk"

    url_obtained = (
        f"https://www.sreality.cz/detail/prodej/"
        f"byt/{proportion}/{flat_locality}/{flat_id}"
    )
    return url_obtained


if __name__ == "__main__":
    logging.basicConfig(
        filename=prediction_config["data_path"] + "final_predictions_training.log",
        level=logging.INFO,
        format="%(asctime)s : %(levelname)s : %(message)s",
    )
    data = pd.read_csv(prediction_config["data_path"])
    flat_ids_to_test = data["ID"].tolist()
    predictions = []
    urls = []
    flat_ids = []

    for flat_id in tqdm(flat_ids_to_test):

        logging.info(f"Making prediction for flat id {flat_id}")
        api_url = prediction_config["api_url"] + str(flat_id)
        try:
            r = requests.post(url=api_url, timeout=5)
            extracted_data = r.json()

            flat_ids.append(flat_id)
            predictions.append(extracted_data["prediction_minus_actual_price"])
            urls.append(reconstruct_url_from_id(flat_id))
            logging.info("ID predicted successfully.")
        except:
            logging.info("Error while predicting of the price.")

    final_data = pd.DataFrame(
        {"flat_id": flat_ids, "prediction_minus_actual": predictions, "url": urls}
    )
    final_data = final_data.sort_values(
        by="prediction_minus_actual", ascending=False
    ).head(FLATS_TO_SEND)
    # Save the predicted data to csv
    final_data.to_csv(prediction_config["data_path"], header=True, index=False)
    # Send them with email
    send_mail(prediction_config["data_path"])
