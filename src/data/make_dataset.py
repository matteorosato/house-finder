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
from sklearn.model_selection import train_test_split

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR.joinpath("data/raw")  # name of the folder for raw data
PROCESSED_DIR = PROJECT_DIR.joinpath("data/processed")  # name of the folder for processed data
MAX_PAGES = 2  # limit of the ads pages to be requested

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())


class Datasource:
    api_key = None
    secret = None

    def __init__(self, name: str, config_filepath: str):
        self.name = name
        self.config_filepath = config_filepath
        self.filtered_params = self.parse_filter_params(
            params_dict=self.read_toml_config(file_path=self.config_filepath))

    @property
    def logger(self):
        return logging.getLogger(f'{__name__}.{self.__class__.__name__}')

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
        output_filename = RAW_DIR.joinpath(f'dump_{int(time.time())}.json')
        self.logger.info(f'Exporting data to {output_filename}')
        with open(output_filename, 'w') as f:
            f.write(json.dumps(results, indent=4))

    def create_dataset(self):
        pass

    def define_search_url(self) -> str:
        pass

    def clean_dataset(self, df) -> pd.DataFrame:
        pass

    @staticmethod
    def create_train_test_df(df, test_size=0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        # split the data into train and test set
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=11, shuffle=True)
        return df_train, df_test


class Idealista(Datasource):
    api_key: str = os.environ['IDEALISTA_API_KEY']
    secret: str = os.environ['IDEALISTA_SECRET']

    def __init__(self, name: str, config_filepath: str):
        super().__init__(name, config_filepath)

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
                        "Content-Type": "application/x-www-form-urlencoded"}
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
            self.logger.info(f"Got {len(elements)} items over a total of {result['total']} available")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error: '{str(e)}'")
            raise

        return result

    def search(self, headers_dict) -> dict:
        r = requests.post(self.search_url, headers=headers_dict, params=self.filtered_params)
        r.raise_for_status()
        result = r.json()
        return result

    def create_dataset(self) -> pd.DataFrame:
        source_dir = RAW_DIR
        json_files = [f for f in pathlib.Path(source_dir).glob("*.json")]
        json_files.reverse()  # put files in descending order
        dfs = []
        for file in json_files:
            with open(file, 'r') as f:
                elements_dict = json.load(f)['elementList']
                dfs.append(pd.DataFrame.from_dict(elements_dict))
        df = pd.concat(dfs)
        return df

    def clean_dataset(self, df) -> pd.DataFrame:

        # removing duplicates based on propertyCode
        duplicates = df.duplicated(subset=['propertyCode'], keep='first')
        self.logger.info(f'Found {sum(duplicates.values)} duplicates. Removing them...')
        df = df[~duplicates.values]

        # keep only specified columns
        columns = ['floor', 'size', 'rooms', 'bathrooms', 'address', 'municipality', 'district', 'status', 'hasLift',
                   'price', 'priceByArea', 'parkingSpace', 'isAuction', 'url', 'propertyCode']
        df = df[columns]

        # fill missing district by setting it equal to municipality
        df.fillna({'district': df['municipality']}, inplace=True)

        # remove auction ads
        df = df[df['isAuction'].isna()]
        df = df.drop(columns=['isAuction'])

        # extract info for parking space
        def has_parking_space(x):
            if isinstance(x, dict) and x['hasParkingSpace'] and x['isParkingSpaceIncludedInPrice']:
                # keep only the ads with park included in the final price
                return True
            else:
                return False

        df['parkingSpace'] = df['parkingSpace'].apply(has_parking_space)

        # fill nan
        df.fillna({'hasLift': False}, inplace=True)

        # cast to int
        cols_to_int = ['price', 'size', 'priceByArea']
        df[cols_to_int] = df[cols_to_int].astype('int')

        # use 'propertyCode' as index
        df.set_index('propertyCode', inplace=True)

        return df

    def build_features(self, df) -> pd.DataFrame:
        # keep only specified columns (features)
        features = ['floor', 'size', 'rooms', 'bathrooms', 'municipality', 'district', 'status', 'hasLift',
                    'parkingSpace', 'price']
        df = df[features]

        # convert to categories
        cols_to_categories = ['floor', 'municipality', 'district', 'status']
        df[cols_to_categories] = df[cols_to_categories].astype('category')

        # apply One Hot Encoding to categories
        df = pd.get_dummies(df, columns=cols_to_categories)

        # checking features which have nan values
        nan_values = df.isna().sum()
        if nan_values.any():
            self.logger.warning("There are still nan values in your dataset. Please check it")

        return df


def main():
    logger = logging.getLogger(__name__)

    idealista = Idealista(name='Idealista', config_filepath='config.toml')
    logger.info(f'Getting results from {idealista.name} website')
    results = idealista.get_results()
    logger.info(f'Exporting results from {idealista.name} website')
    idealista.export_results(results)
    logger.info('Creating dataset...')
    df_raw = idealista.create_dataset()
    logger.info('Cleaning dataset...')
    df_cleaned = idealista.clean_dataset(df_raw)
    logger.info('Exporting cleaned data...')
    df_cleaned.to_csv(PROCESSED_DIR.joinpath('cleaned_data.csv'))
    logger.info('Building features...')
    df_processed = idealista.build_features(df_cleaned)
    logger.info('Separating train and test datasets...')
    df_train, df_test = idealista.create_train_test_df(df_processed)
    logger.info('Exporting train data...')
    df_train.to_csv(PROCESSED_DIR.joinpath('training_data.csv'))
    logger.info('Exporting test data...')
    df_test.to_csv(PROCESSED_DIR.joinpath('test_data.csv'))
    logger.info('Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
