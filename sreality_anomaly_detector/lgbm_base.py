"""Base class for Training and Inference classes."""
import numpy as np
import pandas as pd


class LGBMMBaseModel:
    """Class for prediction of flat prices."""

    def __init__(self):
        """Construct the class."""
        self.numerical_cols = [
            "area",
            "elevator",
            "parking",
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
            "ownership",
            "floor",
            "energy_type",
            "equipped",
        ]
        self.preds = self.numerical_cols + self.categorical_cols
        self.data = pd.DataFrame()

    def retype_data(self):
        """Adjust types of some columns."""
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype("category")

        # Weird columns, that have numeric values, 'true' values and nan values
        for col in ["parking"]:
            self.data[col] = np.where(self.data[col] == "True", "1", self.data[col])
            self.data[col] = self.data[col].astype("float")

        for col in self.numerical_cols:
            self.data[col] = self.data[col].astype("float")
