"""Training class for flat prices prediction."""
import logging
import os
import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from sreality_anomaly_detector.configs import training_config
from sreality_anomaly_detector.lgbm_base import LGBMMBaseModel
from sreality_anomaly_detector.logger import add_logger, close_logger


class LGBMModelTrainer(LGBMMBaseModel):
    """Class for prediction of flat prices."""

    def __init__(self, config):
        """Construct the class."""
        super().__init__()
        self.config = config
        self.target = "price"
        self.params = {
            "early_stopping_rounds": 100,
            "num_boost_round": 10000,
            "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "subsample": 0.7,
            "max_depth": 3,
            "seed": 1234,
            "verbosity": 0,
        }
        self.logger = add_logger(
            os.path.join(config["path_to_save"], "model_training.log")
        )
        self.shap_values = None

    def load_data(self):
        """Load the data."""
        self.data = pd.read_csv(self.config["input_path"])

    def ohe_fit(self):
        """Fit and save one hot encoder."""
        enc = OneHotEncoder(sparse=False)
        enc.fit(self.data[self.categorical_cols])

        pickle.dump(
            enc,
            open(os.path.join(self.config["path_to_save"], "ohe_model.pickle"), "wb"),
        )

    def one_model_lgbm(
        self,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
    ):
        """Fit one LGBM model."""
        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        watchlist = dvalid
        booster = lgb.train(
            params=self.params,
            train_set=dtrain,
            valid_sets=watchlist,
            verbose_eval=20000,
        )

        return booster.best_iteration

    def compute_var_imp(self, model):
        """Compute variable importance."""
        importance_df = pd.DataFrame()
        importance_df["Feature"] = self.preds
        importance_df["Importance_gain"] = model.feature_importance(
            importance_type="gain"
        )

        plt.plot(figsize=(15, 15))
        bar = sns.barplot(
            x="Importance_gain",
            y="Feature",
            data=importance_df.sort_values(by="Importance_gain", ascending=False),
        )
        fig = bar.get_figure()
        fig.savefig(self.config["path_to_save"] + "variable_importance.png")
        logging.info("Variable importance of the features in the model:")
        logging.info(importance_df.to_string())

    def train_model_CV(self):
        """Train cross-validation model. Obtain how many trees should model have."""
        lgbm_rounds = []

        n_splits = 4
        cross_val = KFold(n_splits=n_splits, shuffle=True, random_state=10)

        for train_indexes, valid_indexes in cross_val.split(self.data):
            train_data_lgbm = self.data.iloc[train_indexes][self.preds]
            valid_data_lgbm = self.data.iloc[valid_indexes][self.preds]

            train_target = self.data[self.target].iloc[train_indexes]
            valid_target = self.data[self.target].iloc[valid_indexes]

            lgbm_rounds.append(
                self.one_model_lgbm(
                    train_data_lgbm, valid_data_lgbm, train_target, valid_target
                )
            )

        self.logger.info(f"Optimal number of trees: {int(np.mean(lgbm_rounds))}")
        return int(np.mean(lgbm_rounds))

    def final_model(self, data: pd.DataFrame, target: pd.Series):
        """Fit final model with number of trees estimated from CV."""
        dtrain = lgb.Dataset(data, label=target)
        dvalid = lgb.Dataset(data, label=target)
        booster = lgb.train(
            params=self.params,
            train_set=dtrain,
            valid_sets=dvalid,
        )
        return booster

    def print_and_save_shap_values(self, model: lgb.Booster, ret: bool = False):
        """Compute SHAP values of the model for the data."""
        explainer = shap.TreeExplainer(model)

        pickle.dump(
            explainer,
            open(
                os.path.join(
                    self.config["path_to_save"], "shap_explainer_model.pickle"
                ),
                "wb",
            ),
        )

        shap_values = explainer.shap_values(self.data[self.preds])

        if isinstance(shap_values, list):
            self.shap_values = shap_values[1]
        else:
            self.shap_values = shap_values

        shap.summary_plot(self.shap_values, self.data[self.preds])
        shap.summary_plot(self.shap_values, self.data[self.preds], plot_type="bar")

        if ret:
            return self.shap_values, explainer

    def shap_dependence_plot(self, column: str, interaction_column: str = None):
        """Plot SHAP dependence plot."""
        if interaction_column:
            shap.dependence_plot(
                column,
                self.shap_values,
                self.data[self.preds],
                interaction_index=interaction_column,
            )
        else:
            shap.dependence_plot(column, self.shap_values, self.data[self.preds])

    def fit_and_predict(self):
        """Pipeline that run everything."""
        self.load_data()
        self.preprocess_data()
        if self.config["perform_OHE"]:
            self.ohe_fit()
            self.ohe_predict(
                os.path.join(self.config["path_to_save"], "ohe_model.pickle")
            )
        self.logger.info("Starting oof Cross Validation fit.")
        n_trees = self.train_model_CV()
        # Replace number of iteration for final model
        self.params["num_boost_round"] = n_trees
        self.logger.info("Starting of final model training.")
        final_model = self.final_model(self.data[self.preds], self.data[self.target])
        self.logger.info("Model succesfully trained.")
        self.print_and_save_shap_values(final_model)
        self.data["predictions"] = final_model.predict(self.data[self.preds])
        self.logger.info("R2 score of final model:")
        self.logger.info(r2_score(self.data[self.target], self.data["predictions"]))
        self.compute_var_imp(final_model)

        # Save model
        self.logger.info("Saving of the model.")
        pickle.dump(
            final_model,
            open(os.path.join(self.config["path_to_save"], "lgbm_model.pickle"), "wb"),
        )
        self.logger.info("Model succesfully saved.")
        close_logger(self.logger)


if __name__ == "__main__":
    model = LGBMModelTrainer(training_config)
    model.fit_and_predict()
