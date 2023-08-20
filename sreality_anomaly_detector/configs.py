"""Model config."""
inference_model_config = {"model_path": "/models/lgbm_model.pickle"}

scrape_config = {"data_path": "/data/"}

training_config = {"input_path": "/data/scrape.csv", "path_to_save": "/models/"}

prediction_config = {
    "data_path": "/data/",
    "api_url": "http://localhost:8000/predict?input_data=",
    "model_source": "API",
}
