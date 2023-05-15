"""Script for run of training of LGBMTrainer."""
from sreality_anomaly_detector.lgbm_trainer import LGBMModelTrainer

config = {"input_path": "2023-01-07_scrape.csv", "path_to_save": "lgbm_model.pickle"}

if __name__ == "__main__":
    model = LGBMModelTrainer(config)
    model.fit_and_predict()
