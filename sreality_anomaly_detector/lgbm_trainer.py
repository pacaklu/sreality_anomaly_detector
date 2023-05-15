"""Training class for flat prices prediction."""
import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from lgbm_base import LGBMMBaseModel

# Pycharm setting
desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 20)


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

    def load_data(self):
        """Load the data."""
        self.data = pd.read_csv(self.config["input_path"])

    def one_model_lgbm(self, x_train, x_valid, y_train, y_valid):
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

        print("R2 score on train:")
        print(r2_score(y_train, booster.predict(x_train)))
        print("R2 score on valid")
        print(r2_score(y_valid, booster.predict(x_valid)))

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
        fig.savefig("variable_importance.png")

    def train_model_CV(self):
        """Train cross-validation model. Used for obtaining of how many trees should model have."""
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

        return int(np.mean(lgbm_rounds))

    def final_model(self, data, target):
        """Fit final model with number of trees estimated from CV."""
        dtrain = lgb.Dataset(data, label=target)
        dvalid = lgb.Dataset(data, label=target)
        booster = lgb.train(
            params=self.params,
            train_set=dtrain,
            valid_sets=dvalid,
        )
        return booster

    def fit_and_predict(self):
        """Pipeline that run everything."""
        self.load_data()
        self.retype_data()
        n_trees = self.train_model_CV()
        # Replace number of iteration for final model
        self.params["num_boost_round"] = n_trees
        final_model = self.final_model(self.data[self.preds], self.data[self.target])

        self.data["predictions"] = final_model.predict(self.data[self.preds])
        print("R2 score:")
        print(r2_score(self.data[self.target], self.data["predictions"]))
        self.compute_var_imp(final_model)

        # Save model
        pickle.dump(final_model, open(self.config["path_to_save"], "wb"))
