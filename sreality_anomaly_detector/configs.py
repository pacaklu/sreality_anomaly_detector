"""Model config."""
inference_model_config = {"model_path": "/model/lgbm_model.pickle"}

scrape_config = {"data_path": "./data", "model_save_path": "/model/lgbm_model.pickle"}

training_config = {"input_path": "/data/scrape.csv", "path_to_save": "/models/lgbm_model.pickle"}

prediction_config = {"data_path": "/data/scrape.csv", "api_url": "http://localhost:8000/predict?input_data="}
