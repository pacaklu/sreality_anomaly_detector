"""Predict results for all scraped flats."""
import json
import os

import pandas as pd
import requests
from tqdm import tqdm

from sreality_anomaly_detector.email_sender import send_mail
from sreality_anomaly_detector.configs import inference_model_config, prediction_config
from sreality_anomaly_detector.lgbm_inferor import LGBMModelInferor
from sreality_anomaly_detector.logger import add_logger, close_logger

FLATS_TO_SEND = 20


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


def predict_data_to_all_ids(prediction_config: dict, inference_model_config: dict):
    """Make prediction for flat ids from the config path."""
    logger = add_logger(
        os.path.join(prediction_config["data_path"], "predictions_all.log")
    )
    data = pd.read_csv(os.path.join(prediction_config["data_path"], "scrape.csv"))
    flat_ids_to_test = data["ID"].tolist()
    predictions = []
    urls = []
    flat_ids = []
    top_1_increasing_price_feature = []
    top_2_increasing_price_feature = []
    top_3_increasing_price_feature = []
    top_1_decreasing_price_feature = []
    top_2_decreasing_price_feature = []
    top_3_decreasing_price_feature = []

    logger.info("Making prediction for all flat ids")
    if prediction_config["model_source"] == "LOCAL":
        model = LGBMModelInferor(inference_model_config)

    for flat_id in tqdm(flat_ids_to_test):
        logger.info(f"Making prediction for ID {flat_id}")
        try:
            flat_url = reconstruct_url_from_id(flat_id)
        except KeyError:
            logger.warning(
                f"Prediction unsuccessful for ID {flat_id}. "
                f"impossible to reconstruct url."
            )
            continue

        if prediction_config["model_source"] == "API":
            api_url = prediction_config["api_url"] + str(flat_id)
            logger.info(f"FLAT URL API")
            logger.info(api_url)
            try:
                logger.info(f"sending request to api")
                r = requests.post(url=api_url, timeout=15)
                extracted_data = r.json()
                logger.info(f"received data")
                logger.info(f"extracted_data")
                prediction = extracted_data["prediction_minus_actual_price"]
                flat_ids.append(flat_id)
                predictions.append(prediction)
                urls.append(flat_url)
                top_1_increasing_price_feature.append(extracted_data["top_1_increasing_price_feature"])
                top_2_increasing_price_feature.append(extracted_data["top_2_increasing_price_feature"])
                top_3_increasing_price_feature.append(extracted_data["top_3_increasing_price_feature"])
                top_1_decreasing_price_feature.append(extracted_data["top_1_decreasing_price_feature"])
                top_2_decreasing_price_feature.append(extracted_data["top_2_decreasing_price_feature"])
                top_3_decreasing_price_feature.append(extracted_data["top_3_decreasing_price_feature"])
                logger.info(f"Prediction successful for ID {flat_id}")
            except json.JSONDecodeError:
                logger.warning(f"Prediction unsuccessful for ID {flat_id}")

        else:
            prediction = model.predict(flat_id)
            flat_ids.append(flat_id)
            predictions.append(prediction["prediction_minus_actual_price"])
            urls.append(flat_url)
            top_1_increasing_price_feature.append(prediction["top_1_increasing_price_feature"])
            top_2_increasing_price_feature.append(prediction["top_2_increasing_price_feature"])
            top_3_increasing_price_feature.append(prediction["top_3_increasing_price_feature"])
            top_1_decreasing_price_feature.append(prediction["top_1_decreasing_price_feature"])
            top_2_decreasing_price_feature.append(prediction["top_2_decreasing_price_feature"])
            top_3_decreasing_price_feature.append(prediction["top_3_decreasing_price_feature"])
            logger.info(f"Prediction successful for ID {flat_id}")

    final_data = pd.DataFrame(
        {"flat_id": flat_ids, "prediction_minus_actual": predictions, "url": urls,
         "top_1_increasing_price_feature": top_1_increasing_price_feature,
         "top_1_decreasing_price_feature": top_1_decreasing_price_feature,
         "top_2_increasing_price_feature": top_2_increasing_price_feature,
         "top_2_decreasing_price_feature": top_2_decreasing_price_feature,
         "top_3_increasing_price_feature": top_3_increasing_price_feature,
         "top_3_decreasing_price_feature": top_3_decreasing_price_feature,
         }
    )
    final_data = final_data.sort_values(
        by="prediction_minus_actual", ascending=False
    ).head(FLATS_TO_SEND)
    # Save the predicted data to csv
    final_data.to_csv(
        os.path.join(prediction_config["data_path"], "predictions.csv"),
        header=True,
        index=False,
    )
    # Send them with email
    send_mail(os.path.join(prediction_config["data_path"], "predictions.csv"))
    close_logger(logger)


if __name__ == "__main__":
    predict_data_to_all_ids(prediction_config, inference_model_config)
