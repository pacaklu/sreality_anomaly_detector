"""Base class for Training and Inference classes."""
import pickle

import numpy as np
import pandas as pd


class LGBMMBaseModel:
    """Class for prediction of flat prices."""

    def __init__(self):
        """Construct the class."""
        self.numerical_cols = [
            "area",
            "elevator",
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
        self.categorical_cols = [
            "building",
            "condition",
            "parking",
            "ownership",
            "floor",
            "energy_type",
            "equipped",
        ]
        self.data = pd.DataFrame()

    def preprocess_data(self):
        """Preprocess data."""
        self.data["building"] = np.where(
            self.data["building"] == "Kamenná", "Smíšená", self.data["building"]
        )

        self.data["condition"] = np.where(
            self.data["condition"] == "Před rekonstrukcí",
            "Dobrý",
            self.data["condition"],
        )
        self.data["condition"] = np.where(
            self.data["condition"] == "Projekt", "Novostavba", self.data["condition"]
        )
        self.data["condition"] = np.where(
            self.data["condition"] == "V rekonstrukci",
            "Novostavba",
            self.data["condition"],
        )
        self.data["condition"] = np.where(
            self.data["condition"] == "Špatný", "Dobrý", self.data["condition"]
        )

        self.data["ownership"] = np.where(
            self.data["ownership"] == "Státní/obecní",
            "Družstevní",
            self.data["ownership"],
        )

        self.data["floor"] = np.where(
            self.data["floor"] == "p", "0", self.data["floor"]
        )
        self.data["floor"] = np.where(
            self.data["floor"] == "5", ">4", self.data["floor"]
        )
        self.data["floor"] = np.where(
            self.data["floor"] == "6", ">4", self.data["floor"]
        )
        self.data["floor"] = np.where(
            self.data["floor"] == "7", ">4", self.data["floor"]
        )
        self.data["floor"] = np.where(
            self.data["floor"] == "8", ">4", self.data["floor"]
        )
        self.data["floor"] = np.where(
            self.data["floor"] == "9", ">4", self.data["floor"]
        )
        self.data["floor"] = np.where(
            self.data["floor"] == "10", ">4", self.data["floor"]
        )

        self.data["parking"] = np.where(
            ((self.data["parking"] == "True") | (self.data["parking"].isnull())),
            self.data["parking"],
            "1",
        )

        for col in self.categorical_cols:
            self.data[col] = np.where(self.data[col].isnull(), "NAN", self.data[col])
            self.data[col] = self.data[col].astype("category")

        for col in self.numerical_cols:
            self.data[col] = self.data[col].astype("float")
            self.preds = self.numerical_cols + self.categorical_cols

    def ohe_predict(self, ohe_model_path):
        """Predict from ohe model."""
        with open(ohe_model_path, "rb") as file:
            ohe_model = pickle.load(file)

        x = ohe_model.transform(self.data[self.categorical_cols])
        ohe_df = pd.DataFrame(
            x, columns=ohe_model.get_feature_names(input_features=self.categorical_cols)
        )
        self.data = pd.concat([self.data, ohe_df], axis=1)
        self.preds = self.numerical_cols + list(ohe_df.columns)
