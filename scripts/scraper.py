"""Class for scraping of flats from Sreality."""
import logging
import math
import os
from typing import Optional

import pandas as pd
import requests
from sreality_anomaly_detector.configs import scrape_config
from sreality_anomaly_detector.lgbm_inferor import extract_one_flat_details
from sreality_anomaly_detector.logger import add_logger, close_logger
from tqdm import tqdm

# Available is 20, 40, 60
SCRAPE_FLATS_PER_PAGE = 60


class SrealityScraper:
    """Main class for scraping."""

    def __init__(self, scrape_config: dict):
        """Initialize of parameters."""
        self.config = scrape_config
        self.number_of_pages_to_scrap = None
        self.list_of_flat_ids = None
        self.logger = add_logger(
            os.path.join(scrape_config["data_path"], "scraping.log")
        )

    def count_number_of_pages_needed(self):
        """Obtain how many pages we have to scrape."""
        # API request for obtaining of count of flats on Sreality.cz
        # Documentation is here, on page 15
        # https://dspace.cvut.cz/bitstream/handle/10467/103384/F8-BP-2021-Malach-Ondrej-thesis.pdf?sequence=-1
        url = (
            "https://www.sreality.cz/api/cs/v2/estates/count?category_sub_cb=4|5&"
            "category_main_cb=1&locality_region_id=10&category_type_cb=1"
        )
        obtained_json = requests.get(url=url)
        obtained_json = obtained_json.json()
        number_of_available_flats = int(obtained_json["result_size"])

        self.number_of_pages_to_scrap = math.ceil(
            number_of_available_flats / SCRAPE_FLATS_PER_PAGE
        )

    def obtain_ids_of_all_available_flats(self):
        """Collect ids of all available flats."""
        list_of_flat_ids = []
        for page_number in tqdm(range(self.number_of_pages_to_scrap)):
            url = (
                f"https://www.sreality.cz/api/cs/v2/"
                f"estates?category_sub_cb=4|5&category_main_cb=1&"
                f"locality_region_id=10&category_type_cb=1&per_page="
                f"{SCRAPE_FLATS_PER_PAGE}&page={page_number}"
            )
            obtained_json = requests.get(url=url)
            obtained_json = obtained_json.json()
            for index in range(SCRAPE_FLATS_PER_PAGE):
                flat_id = obtained_json["_embedded"]["estates"][index]["_embedded"][
                    "favourite"
                ]["_links"]["self"]["href"]
                # flat_id is in format /cs/v2/favourite/ID
                flat_id = flat_id.split("/")[-1]
                list_of_flat_ids.append(flat_id)

        # Remove duplicates
        self.list_of_flat_ids = list(set(list_of_flat_ids))

    @staticmethod
    def request_one_flat(flat_id: str) -> Optional[dict]:
        """Request API with 1 flat id and return response."""
        url = f"https://www.sreality.cz/api/cs/v2/estates/{flat_id}"
        try:
            obtained_json = requests.get(url=url, timeout=5)
            obtained_json = obtained_json.json()
        except requests.exceptions.Timeout:
            return {}

        return obtained_json

    def create_and_save_df_with_data(self):
        """Create and save dataframe with all scraped flats."""
        list_of_dicts = []
        list_of_valid_flat_ids = []
        for flat_id in tqdm(self.list_of_flat_ids[:100]):
            self.logger.info(f"Processing flat ID {flat_id}")
            flat_api_response = self.request_one_flat(flat_id)
            one_flat_details = extract_one_flat_details(flat_api_response)

            if one_flat_details:
                list_of_dicts.append(one_flat_details)
                list_of_valid_flat_ids.append(flat_id)
            self.logger.info(f"Flat ID {flat_id} processed.")

        dataframe = pd.DataFrame(list_of_dicts)
        dataframe["ID"] = list_of_valid_flat_ids
        self.logger.info("Creating scraped Dataframe and saving.")
        dataframe.to_csv(
            os.path.join(self.config["data_path"], "scrape.csv"),
            header=True,
            index=False,
        )
        self.logger.info("Data succesfully saved. ")
        close_logger(self.logger)

    def scrape_pipeline(self):
        """One function that wraps all steps."""
        logging.info("Starting whole scraping process.")
        self.count_number_of_pages_needed()
        self.obtain_ids_of_all_available_flats()
        self.create_and_save_df_with_data()


if __name__ == "__main__":
    scraper = SrealityScraper(scrape_config)
    scraper.scrape_pipeline()
