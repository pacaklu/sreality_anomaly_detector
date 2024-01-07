"""Model config."""
inference_model_config = {
    "model_path": "/models/lgbm_model.pickle",
    "shap_explainer_model_path": "/models/shap_explainer_model.pickle",
    "filter_query": "price < 7000000 and (condition == 'Velmi dobrý' | condition == 'Novostavba')",
    "perform_OHE": False,
}

scrape_config = {"data_path": "/data/"}

training_config = {
    "input_path": "/data/scrape.csv",
    "path_to_save": "/models/",
    "perform_OHE": False,
}

prediction_config = {
    "data_path": "/data/",
    "api_url": "http://localhost:8000/predict?input_data=",
    "model_source": "LOCAL", # LOCAL or API available
}
