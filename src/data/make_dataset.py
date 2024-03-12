# -*- coding: utf-8 -*-
import base64
import json
import logging
import os
import pathlib
import time
import pandas as pd
import requests
import toml
from dotenv import find_dotenv, load_dotenv

RAW = 'raw'  # name of the folder for raw data
PROCESSED = 'processed'  # name of the folder for processed data
MAX_PAGES = 2  # limit of the ads pages to be requested

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())


class Datasource:
    api_key = None
    secret = None

    logger = logging.getLogger(__name__)

    def __init__(self, name: str, config_filepath: str, data_dir: str):
        self.name = name
        self.config_filepath = config_filepath
        self.data_dir = data_dir
        self.df = None

    @property
    def filtered_params(self):
        return self.parse_filter_params(
            params_dict=self.read_toml_config(file_path=self.config_filepath))

    @property
    def search_url(self):
        return self.define_search_url()

    @staticmethod
    def read_toml_config(file_path: str) -> dict:
        with open(file_path, 'r') as file:
            config_dict = toml.load(file)
        return config_dict

    @staticmethod
    def parse_filter_params(params_dict: dict) -> dict:
        filtered_params = dict()
        for dictionary in params_dict.values():
            for k, v in dictionary.items():
                if str(v):  # keep non-empty values only
                    filtered_params[k] = v
        return filtered_params

    def get_oauth_token(self) -> str:
        pass

    def get_results(self) -> dict:
        pass

    def export_results(self, results: dict):
        # export results from query
        output_filename = os.path.join(self.data_dir, RAW, f'dump_{int(time.time())}.json')
        self.logger.info(f'Exporting data to {output_filename}')
        with open(output_filename, 'w') as f:
            f.write(json.dumps(results, indent=4))

    def create_dataset(self):
        pass

    def define_search_url(self) -> str:
        pass

    def clean_dataset(self):
        pass

    def export_dataset(self):
        # export df for backup purposes
        output_filename = os.path.join(self.data_dir, PROCESSED, f'df_total_{int(time.time())}.csv')
        self.df.to_csv(output_filename)


class Idealista(Datasource):
    api_key: str = os.environ['IDEALISTA_API_KEY']
    secret: str = os.environ['IDEALISTA_SECRET']

    def __init__(self, name: str, config_filepath: str, data_dir: str):
        super().__init__(name, config_filepath, data_dir)

    def define_search_url(self) -> str:
        country = self.filtered_params['country']
        search_url = f'https://api.idealista.com/3.5/{country}/search'
        return search_url

    def get_oauth_token(self) -> str:
        message = f"{self.api_key}:{self.secret}"

        # deal with bytes-like object
        message_bytes = message.encode('ascii')
        base64_bytes = base64.b64encode(message_bytes)
        base64_message = base64_bytes.decode('ascii')

        auth_header = f"Basic {base64_message}"

        headers_dict = {"Authorization": auth_header,
                        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}

        params_dict = {"grant_type": "client_credentials",
                       "scope": "read"}

        try:
            r = requests.post("https://api.idealista.com/oauth/token",
                              headers=headers_dict,
                              params=params_dict)
            r.raise_for_status()
            token = r.json()["access_token"]
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error: '{str(e)}'")
            raise

        return token

    def get_results(self) -> dict:
        token = self.get_oauth_token()
        headers_dict = {"Authorization": 'Bearer ' + token,
                        "Content-Type": "application/x-www-form-urlencoded"
                        }
        try:
            elements = []
            result = self.search(headers_dict)  # get results for the first page
            self.logger.info(f"Available items: {result['total']} ({result['totalPages']} pages)")
            elements.extend(result["elementList"])

            for i in range(2, min(MAX_PAGES, result["totalPages"]) + 1):
                self.filtered_params["numPage"] = i
                result = self.search(headers_dict)  # get results for the subsequent pages
                elements.extend(result["elementList"])

            result["elementList"] = elements  # update dictionary with cumulative results
            self.logger.info(f"Stored {len(elements)} items over a total of {result['total']} available")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error: '{str(e)}'")
            raise

        return result

    def search(self, headers_dict) -> dict:
        r = requests.post(self.search_url, headers=headers_dict, params=self.filtered_params)
        r.raise_for_status()
        result = r.json()
        return result

    def create_dataset(self):
        source_dir = os.path.join(self.data_dir, RAW)
        json_files = [f for f in pathlib.Path(source_dir).glob("*.json")]
        json_files.reverse()  # put files in descending order
        dfs = []
        for file in json_files:
            with open(file, 'r') as f:
                elements_dict = json.load(f)['elementList']
                dfs.append(pd.DataFrame.from_dict(elements_dict))
        self.df = pd.concat(dfs)
        # removing duplicates based on propertyCode
        self.df.drop_duplicates(subset=['propertyCode'], keep='first')

    def clean_dataset(self):
        columns = ['propertyCode', 'floor', 'price', 'size', 'rooms', 'bathrooms', 'address', 'province',
                   'municipality', 'district', 'latitude', 'longitude', 'showAddress', 'url', 'distance', 'description',
                   'status', 'newDevelopment', 'hasLift', 'priceByArea', 'detailedType', 'hasPlan', 'hasStaging',
                   'topNewDevelopment', 'topPlus', 'externalReference', 'isAuction', 'parkingSpace', 'labels',
                   'highlight', 'newDevelopmentFinished']
        self.df = self.df[columns]  # keep only specified columns

        # cast to int
        self.df = self.df.astype({'price': 'int', 'size': 'int', 'priceByArea': 'int'})

        # remove auction ads
        self.df = self.df[self.df['isAuction'].isna()]
        self.df = self.df.drop(columns=['isAuction'])

        # convert floors to numbers
        self.df['floor'] = self.df['floor'].replace('ss', -1).replace('bj', 0).replace('en', 0.5)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    idealista = Idealista(name='Idealista', config_filepath='config.toml', data_dir='data')
    logger.info(f'Getting results from {idealista.name} website')
    results = idealista.get_results()
    logger.info(f'Exporting results from {idealista.name} website')
    idealista.export_results(results)
    logger.info('Creating dataset...')
    idealista.create_dataset()
    logger.info('Cleaning dataset...')
    idealista.clean_dataset()
    logger.info(f'Exporting dataset...')
    idealista.export_dataset()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
