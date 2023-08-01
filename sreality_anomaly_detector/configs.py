"""Model config."""
inference_model_config = {"model_path": "/models/lgbm_model.pickle"}

scrape_config = {"data_path": "/data/scrape.csv"}

training_config = {"input_path": "/data/scrape.csv", "path_to_save": "/models/"}

prediction_config = {"data_path": "/data/scrape.csv", "api_url": "http://localhost:8000/predict?input_data=", "predicted_data_path" : "/data/final_predictions.csv"}
