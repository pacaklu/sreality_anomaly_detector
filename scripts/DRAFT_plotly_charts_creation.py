"""Draft of the script for charts of the prices development. Not used so far."""
import glob

import numpy as np
import pandas as pd
import plotly
import plotly.express as px


class ChartGenerator:
    """Class for creation of plotly charts."""

    def __init__(self, input_path: str):
        """Initialize parameters."""
        self.html_page = None
        self.input_path = input_path
        self.list_of_dataframes = []
        self.list_of_names = []
        self.list_of_html_divs = []

    def render_html_site(self):
        """Render html site with created charts."""
        # Initialize page
        html_page = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            "<title>Plotly Charts</title></head><body>"
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        )

        for div in self.list_of_html_divs:
            html_page = html_page + div

        # finish page
        html_page = html_page + "</body></html>"

        file = open("first_attempt.html", "w")
        file.write(html_page)
        file.close()

    def load_data(self):
        """Load all data from input path and extract the names from it."""
        for path in glob.glob(self.input_path + "*.csv"):
            self.list_of_dataframes.append(pd.read_csv(path))
            self.list_of_names.append(path.split("\\")[-1].split("_")[0])

    def create_chart_mean(self):
        """Create chart mean of price per apartment."""
        mean_of_price_per_apartment = []
        for df in self.list_of_dataframes:
            mean_of_price_per_apartment.append(np.nanmean(df["price"]))

        fig = px.line(
            x=self.list_of_names,
            y=mean_of_price_per_apartment,
            title="Mean price of flat.",
        )
        self.list_of_html_divs.append(
            plotly.offline.plot(fig, include_plotlyjs=False, output_type="div")
        )

    def create_chart_mean_per_1m(self):
        """Create chart mean of price per apartment."""
        mean_of_price_per_1m2 = []
        for df in self.list_of_dataframes:
            mean_of_price_per_1m2.append(np.nansum(df["price"]) / np.nansum(df["area"]))

        fig = px.line(
            x=self.list_of_names,
            y=mean_of_price_per_1m2,
            title="Mean price squared meter.",
        )
        self.list_of_html_divs.append(
            plotly.offline.plot(fig, include_plotlyjs=False, output_type="div")
        )


if __name__ == "__main__":
    input_path = r"C:\\Users\\pacak\\PycharmProjects\\pythonProject\\"
    generator = ChartGenerator(input_path)
    generator.load_data()
    generator.create_chart_mean()
    generator.create_chart_mean_per_1m()
    generator.render_html_site()
