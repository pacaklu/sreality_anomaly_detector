"""Inference class for prediction of flat price."""
import logging
import pickle
from typing import Optional

import geopy.distance
import numpy as np
import pandas as pd
import requests

from sreality_anomaly_detector.lgbm_base import LGBMMBaseModel

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
        if item["type"] == "energy_efficiency_rating":
            dict_of_info["energy_type"] = item["value_type"]
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

    if "poi" in obtained_json.keys():
        for poi in obtained_json["poi"]:
            if poi["name"] == "Vlak":
                dict_of_info["train_distance"] = _try_poi_compute(poi)
            if poi["name"] == "Metro":
                dict_of_info["metro_distance"] = _try_poi_compute(poi)
            if poi["name"] == "Restaurace":
                dict_of_info["restaurant_distance"] = _try_poi_compute(poi)

        try:
            dict_of_info["closest_transport_distance"] = obtained_json["poi_transport"]["values"][0]["distance"]
        except KeyError:
            dict_of_info["closest_transport_distance"] = np.NaN
        try:
            dict_of_info["closest_shop_distance"] = obtained_json["poi_grocery"]["values"][0]["distance"]
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
        self.model = None

    def _load_model(self):
        """Load LGBM model from path."""
        with open(self.config["model_path"], "rb") as file:
            self.model = pickle.load(file)

    def _request_flat_data(self, input_flat_id: int) -> dict:
        """Request Sreality api with flat id to receive data."""
        url = f"https://www.sreality.cz/api/cs/v2/estates/{input_flat_id}"
        obtained_json = requests.get(url=url).json()
        return obtained_json

    def predict(self, input_flat_id):
        """Predict price of the flat."""
        try:
            logging.warning("Loading Model")
            self._load_model()
            logging.warning("Model successfully loaded")

            logging.warning("Trying to request data from Sreality API")
            obtained_json = self._request_flat_data(input_flat_id)
            logging.warning("Data successfully requested")

            preprocessed_data = extract_one_flat_details(obtained_json)
            self.data = pd.DataFrame(preprocessed_data, index=[0])
            
            self.retype_data()
            logging.warning("Data successfully preprocessed")
            prediction = self.model.predict(self.data[self.preds])
            prediction_minus_actual = prediction[0] - self.data["price"][0]
        except:
            prediction_minus_actual = float("nan")

        return {"flat_id": input_flat_id, "prediction_minus_actual_price": prediction_minus_actual}
