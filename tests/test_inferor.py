"""Tests for Training and inference classes."""
# Put home directory to the output path
import json
import sys

import pandas as pd

sys.path.append(r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector")

from sreality_anomaly_detector.lgbm_inferor import (LGBMModelInferor,
                                            distance_from_centre,
                                            extract_one_flat_details)

model_config = {
    "model_path": r"C:\Users\pacak\PycharmProjects\sreality_detector\model\lgbm_model.pickle"
}
TESTING_JSON_PATH = (
    r"C:\Users\pacak\PycharmProjects\sreality_detector\tests\testing_flat_data.json"
)


def test_distance_from_centre():
    """Test distance from the mean method."""
    f = open(TESTING_JSON_PATH)
    obtained_json = json.load(f)
    distance = distance_from_centre(obtained_json)
    assert distance == 1.9262761876453975


def test_extract_one_flat_details():
    """Test extract one flat details method."""
    f = open(TESTING_JSON_PATH)
    obtained_json = json.load(f)
    extracted_data = extract_one_flat_details(obtained_json)
    assert extracted_data == {
        "building": "Cihlová",
        "condition": "Po rekonstrukci",
        "ownership": "Osobní",
        "floor": "4",
        "area": "64",
        "energy_type": "D",
        "elevator": True,
        "parking": None,
        "barrierless": None,
        "price": 11425000,
        "equipped": True,
        "closest_transport_distance": 139.0,
        "closest_shop_distance": 259.0,
        "metro_distance": 268,
        "train_distance": 1882,
        "restaurant_distance": 81,
        "distance_to_centre": 1.9262761876453975,
        "balcony": True,
        "cellar": None,
    }


def test_constructor():
    """Test constructor of LGBMModelInferor class."""
    model = LGBMModelInferor(model_config)
    assert model.numerical_cols == [
        "area",
        "elevator",
        "parking",
        "barrierless",
        "closest_transport_distance",
        "closest_shop_distance",
        "metro_distance",
        "train_distance",
        "restaurant_distance",
        "distance_to_centre",
        "balcony",
        "cellar",
    ]
    assert model.categorical_cols == [
        "building",
        "condition",
        "ownership",
        "floor",
        "energy_type",
        "equipped",
    ]
    pd.testing.assert_frame_equal(model.data, pd.DataFrame())

def test_request_flat_data():
    """Test whether request for flat returns proper results."""
    model = LGBMModelInferor(model_config)
    flat_id_to_test = 4065768524
    obtained_json = model._request_flat_data(flat_id_to_test)
    assert len(obtained_json) >0
    assert type(obtained_json) == dict

def test_load_model():
    """Test _load method."""
    model = LGBMModelInferor(model_config)
    model._load_model()
    # Check whether model is properly loaded by having
    # model.model.best_iteration > 0
    assert model.model.best_iteration > 0


def test_predict():
    """Test predict method of LGBMModelInferor class."""
    model = LGBMModelInferor(model_config)
    flat_id_to_test = 4065768524
    result = model.predict(flat_id_to_test)
    assert result["flat_id"] == flat_id_to_test
    assert result["prediction"] > 0
