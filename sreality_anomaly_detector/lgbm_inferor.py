"""Inference class for prediction of flat price."""
import pickle
from typing import Optional

import geopy.distance
import numpy as np
import pandas as pd
import requests

from sreality_anomaly_detector.lgbm_base import LGBMMBaseModel
from sreality_anomaly_detector.logger import add_logger


# Coordinates of prague city centre
CENTRE_COORD = (50.082164, 14.426307)


def _try_poi_compute(poi: dict) -> float:
    """Try to compute walkdistance to given poi."""
    try:
        return poi["walkDistance"]
    except KeyError:
        return np.NaN


def extract_one_flat_details(obtained_json: dict) -> Optional[dict]:
    """Extract flat details from the dictionary."""
    # Check if "price_czk" not in keys of received json
    if "price_czk" not in obtained_json.keys():
        return

    # Check if it contains price and if not, exit
    if "value_raw" not in obtained_json["price_czk"]:
        return

    dict_of_info = {}

    for key in [
        "building",
        "condition",
        "ownership",
        "floor",
        "area",
        "energy_type",
        "elevator",
        "parking",
        "barrierless",
        "price",
        "equipped",
        "closest_transport_distance",
        "closest_shop_distance",
        "metro_distance",
        "train_distance",
        "restaurant_distance",
        "distance_to_centre",
        "balcony",
        "cellar",
    ]:
        dict_of_info[key] = None

    dict_of_info["price"] = obtained_json["price_czk"]["value_raw"]

    for item in obtained_json["items"]:
        if "name" in item.keys():
            if item["name"] == "Stavba":
                dict_of_info["building"] = item["value"]
            if item["name"] == "Stav objektu":
                dict_of_info["condition"] = item["value"]
            if item["name"] == "Vlastnictví":
                dict_of_info["ownership"] = item["value"]
            if item["name"] == "Podlaží":
                dict_of_info["floor"] = item["value"][0]
            if item["name"] == "Užitná plocha":
                dict_of_info["area"] = item["value"]
            if item["name"] == "Výtah":
                dict_of_info["elevator"] = item["value"]
            if item["name"] == "Parkování":
                dict_of_info["parking"] = item["value"]
            if item["name"] == "Bezbariérový":
                dict_of_info["barrierless"] = item["value"]
            if item["name"] == "Vybavení":
                dict_of_info["equipped"] = item["value"]
            if item["name"] == "Lodžie":
                dict_of_info["balcony"] = True
            if item["name"] == "Terase":
                dict_of_info["balcony"] = True
            if item["name"] == "Balkón":
                dict_of_info["balcony"] = True
            if item["name"] == "Sklep":
                dict_of_info["cellar"] = True
        if "type" in item.keys():
            if item["type"] == "energy_efficiency_rating":
                dict_of_info["energy_type"] = item["value_type"]

    if "poi" in obtained_json.keys():
        for poi in obtained_json["poi"]:
            if poi["name"] == "Vlak":
                dict_of_info["train_distance"] = _try_poi_compute(poi)
            if poi["name"] == "Metro":
                dict_of_info["metro_distance"] = _try_poi_compute(poi)
            if poi["name"] == "Restaurace":
                dict_of_info["restaurant_distance"] = _try_poi_compute(poi)

        try:
            dict_of_info["closest_transport_distance"] = obtained_json["poi_transport"][
                "values"
            ][0]["distance"]
        except KeyError:
            dict_of_info["closest_transport_distance"] = np.NaN
        try:
            dict_of_info["closest_shop_distance"] = obtained_json["poi_grocery"][
                "values"
            ][0]["distance"]
        except KeyError:
            dict_of_info["closest_shop_distance"] = np.NaN

    dict_of_info["distance_to_centre"] = distance_from_centre(obtained_json)

    return dict_of_info


def distance_from_centre(obtained_json: dict) -> float:
    """Compute distance from Prague's city centre to actual flat."""
    lat = float(obtained_json["map"]["lat"])
    lon = float(obtained_json["map"]["lon"])
    flat_coord = (lat, lon)

    return geopy.distance.geodesic(flat_coord, CENTRE_COORD).km


class LGBMModelInferor(LGBMMBaseModel):
    """Class for prediction of flat prices."""

    def __init__(self, config):
        """Initialize class."""
        super().__init__()
        self.config = config
        self.logger = add_logger("/data/predicting.log")
        self.model = None
        self.shap_explainer_model = None
        self.shap_values = None
        with open(self.config["model_path"], "rb") as file:
            self.model = pickle.load(file)
        with open(self.config["shap_explainer_model_path"], "rb") as file:
            self.shap_explainer_model = pickle.load(file)


    def _request_flat_data(self, input_flat_id: int) -> dict:
        """Request Sreality api with flat id to receive data."""
        url = f"https://www.sreality.cz/api/cs/v2/estates/{input_flat_id}"
        obtained_json = requests.get(url=url, timeout=5).json()
        return obtained_json

    def predict_shap_values(self):
        """Predict shap values of for the given data."""
        shap_values = self.shap_explainer_model(self.data[self.preds])

        if isinstance(shap_values, list):
            self.shap_values = shap_values[1]
        else:
            self.shap_values = shap_values

        self.shap_values = pd.DataFrame(self.shap_values.values)

        def extract_increasing_price_col(row, pos):
            """Extract feature that is increasing the price on position pos."""
            pos_values = []
            for i in range(len(self.preds)):
                if row[i] > 0:
                    pos_values.append(row[i])

            pos_values.sort(reverse=True)
            try:
                des_val = pos_values[(pos - 1)]
                des_column = self.preds[list(row).index(des_val)]
            except IndexError:
                return "None"

            return {des_column: des_val}

        def extract_decreasing_price_col(row, pos):
            """Extract feature that is increasing the price on position pos."""
            neg_values = []
            for i in range(len(self.preds)):
                if row[i] < 0:
                    neg_values.append(abs(row[i]))

            neg_values.sort(reverse=True)
            try:
                des_val = neg_values[(pos - 1)] * (-1)
                des_column = self.preds[list(row).index(des_val)]
            except IndexError:
                return "None"

            return {des_column: des_val}

        for val in [1, 2, 3]:
            self.shap_values[
                f"top_{val}_increasing_price_feature"
            ] = self.shap_values.apply(
                extract_increasing_price_col, args=(val,), axis=1
            )
            self.shap_values[
                f"top_{val}_decreasing_price_feature"
            ] = self.shap_values.apply(
                extract_decreasing_price_col, args=(val,), axis=1
            )

    def predict(self, input_flat_id: int):
        """Predict price of the flat."""
        self.logger.info('STARTING PREDICTION IN INFEROR')
        obtained_json = self._request_flat_data(input_flat_id)
        self.logger.info('OBTAINED JSOn')
        self.logger.info(obtained_json)
        preprocessed_data = extract_one_flat_details(obtained_json)
        self.logger.info('preprocessed_data')
        self.logger.info(preprocessed_data)
        self.data = pd.DataFrame(preprocessed_data, index=[0])
        try:
            self.preprocess_data()
            self.logger.info(self.data)
            if self.config["perform_OHE"]:
                self.ohe_predict(self.config["ohe_model_path"])
        except KeyError:
            return {
                "flat_id": input_flat_id,
                "prediction_minus_actual_price": float("nan"),
                "top_1_increasing_price_feature": float("nan"),
                "top_1_decreasing_price_feature": float("nan"),
                "top_2_increasing_price_feature": float("nan"),
                "top_2_decreasing_price_feature": float("nan"),
                "top_3_increasing_price_feature": float("nan"),
                "top_3_decreasing_price_feature": float("nan"),
            }

        if self.config["filter_query"]:
            self.data = self.data.query(self.config["filter_query"])

        if len(self.data) > 0:
            prediction = self.model.predict(self.data[self.preds])
            self.predict_shap_values()
            prediction_minus_actual = prediction[0] - self.data["price"][0]
            top_1_increasing_price_feature = self.shap_values[
                "top_1_increasing_price_feature"
            ].iloc[0]
            top_1_decreasing_price_feature = self.shap_values[
                "top_1_decreasing_price_feature"
            ].iloc[0]
            top_2_increasing_price_feature = self.shap_values[
                "top_2_increasing_price_feature"
            ].iloc[0]
            top_2_decreasing_price_feature = self.shap_values[
                "top_2_decreasing_price_feature"
            ].iloc[0]
            top_3_increasing_price_feature = self.shap_values[
                "top_3_increasing_price_feature"
            ].iloc[0]
            top_3_decreasing_price_feature = self.shap_values[
                "top_3_decreasing_price_feature"
            ].iloc[0]
        else:
            prediction_minus_actual = float("nan")
            top_1_increasing_price_feature = float("nan")
            top_1_decreasing_price_feature = float("nan")
            top_2_increasing_price_feature = float("nan")
            top_2_decreasing_price_feature = float("nan")
            top_3_increasing_price_feature = float("nan")
            top_3_decreasing_price_feature = float("nan")
        return {
            "flat_id": input_flat_id,
            "prediction_minus_actual_price": prediction_minus_actual,
            "top_1_increasing_price_feature": top_1_increasing_price_feature,
            "top_1_decreasing_price_feature": top_1_decreasing_price_feature,
            "top_2_increasing_price_feature": top_2_increasing_price_feature,
            "top_2_decreasing_price_feature": top_2_decreasing_price_feature,
            "top_3_increasing_price_feature": top_3_increasing_price_feature,
            "top_3_decreasing_price_feature": top_3_decreasing_price_feature,
        }
