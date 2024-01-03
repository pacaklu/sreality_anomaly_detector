"""Tests for Training and inference classes."""
import sys

sys.path.append(r"/")

from sreality_anomaly_detector.lgbm_trainer import LGBMModelTrainer  # noqa
from sreality_anomaly_detector.predict_all_ids import \
    predict_data_to_all_ids  # noqa
from sreality_anomaly_detector.scraper import SrealityScraper  # noqa

local_training_config = {
    "input_path": (
        r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector"
        r"\data\data_example.csv"
    ),
    "path_to_save": r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector\models",
    "perform_OHE": False,
}
local_scrape_config = {
    "data_path": (r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector\data")
}
local_prediction_config = {
    "data_path": r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector\data",
    "model_source": "local",
}
local_inference_model_config = {
    "model_path": (
        r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector"
        r"\models\lgbm_model.pickle"),
    "ohe_model_path": (
        r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector"
        r"\models\ohe_model.pickle"),
    "shap_explainer_model_path": r"C:\Users\pacak\PycharmProjects\sreality_anomaly_detector\models\shap_explainer_model.pickle",
    "perform_OHE": False,
    "filter_query": "price < 6000000 and floor!='-1'"
}

RUN_LOCAL_SCRAPER = True
RUN_LOCAL_TRAINING = False
RUN_LOCAL_PREDICTIONS = False


# Local Scrape of data
def run_local_scrape():
    """Run scraping locally."""
    scraper = SrealityScraper(local_scrape_config)
    scraper.scrape_pipeline()


# Local Model Training
def run_local_training():
    """Run training of model locally."""
    model = LGBMModelTrainer(local_training_config)
    model.fit_and_predict()


def run_local_predictions():
    """Run predictions of the model locally."""
    predict_data_to_all_ids(local_prediction_config, local_inference_model_config)


if __name__ == "__main__":
    if RUN_LOCAL_SCRAPER:
        run_local_scrape()
    if RUN_LOCAL_TRAINING:
        run_local_training()
    if RUN_LOCAL_PREDICTIONS:
        run_local_predictions()
