"""Model config."""
inference_model_config = {"model_path": "/models/lgbm_model.pickle",
                          "filter_query": "price < 6000000 and floor!='-1'"}

scrape_config = {"data_path": "/data/"}

training_config = {"input_path": "/data/scrape.csv", "path_to_save": "/models/"}

prediction_config = {
    "data_path": "/data/",
    "api_url": "http://localhost:8000/predict?input_data=",
    "model_source": "API",
}
