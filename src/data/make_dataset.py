# -*- coding: utf-8 -*-
import base64
import json
import os
import pathlib
import time

import click
import logging
from pathlib import Path

import pandas as pd
import requests
import toml
from dotenv import find_dotenv, load_dotenv


def create_dataset(source_dir) -> pd.DataFrame:
    json_files = [f for f in pathlib.Path(source_dir).glob("*.json")]
    json_files.reverse()  # put files in descending order
    dfs = []
    for file in json_files:
        with open(file, 'r') as f:
            elements_dict = json.load(f)['elementList']
            dfs.append(pd.DataFrame.from_dict(elements_dict))
    merged_df = pd.concat(dfs)
    merged_df = merged_df.drop_duplicates(subset=['propertyCode'], keep='first')
    return merged_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['propertyCode', 'floor', 'price', 'size', 'rooms', 'bathrooms', 'address', 'province', 'municipality',
               'district', 'latitude', 'longitude', 'showAddress', 'url', 'distance', 'description', 'status',
               'newDevelopment', 'hasLift', 'priceByArea', 'detailedType', 'hasPlan', 'hasStaging', 'topNewDevelopment',
               'topPlus', 'externalReference', 'isAuction', 'parkingSpace', 'labels', 'highlight',
               'newDevelopmentFinished']
    df = df[columns]  # keep only specified columns

    # cast to int
    df = df.astype({'price': 'int', 'size': 'int', 'priceByArea': 'int'})

    # remove auction ads
    df = df[df['isAuction'].isna()]
    df = df.drop(columns=['isAuction'])

    # convert floors to numbers
    df['floor'] = df['floor'].replace('ss', -1).replace('bj', 0).replace('en', 0.5)

    return df


def export_dataset(df, output_dir):
    # export df for backup purposes
    output_filename = os.path.join(output_dir, f'df_total_{int(time.time())}.csv')
    df.to_csv(output_filename)


def get_oauth_token(api_key: str, secret: str) -> str:
    message = f"{api_key}:{secret}"

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
        print(f"Connection error: '{str(e)}'")
        raise

    return token


def read_toml_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config_dict = toml.load(file)
    return config_dict


def parse_filter_params(params_dict: dict) -> dict:
    filtered_params = dict()
    for dictionary in params_dict.values():
        for k, v in dictionary.items():
            if str(v):  # keep non-empty values only
                filtered_params[k] = v
    return filtered_params


def define_search_url(country: str) -> str:
    search_url = f'https://api.idealista.com/3.5/{country}/search'
    return search_url


def get_results(url, params) -> dict:
    token = get_oauth_token(IDEALISTA_API_KEY, IDEALISTA_SECRET)
    headers_dict = {"Authorization": 'Bearer ' + token,
                    "Content-Type": "application/x-www-form-urlencoded"
                    }
    try:
        r = requests.post(url, headers=headers_dict, params=params)
        r.raise_for_status()

        result = r.json()
    except requests.exceptions.RequestException as e:
        print(f"Connection error: '{str(e)}'")
        raise

    return result


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    config_filepath = 'config.toml'
    params = read_toml_config(config_filepath)
    filtered_params = parse_filter_params(params)
    url = define_search_url(country=filtered_params['country'])
    logger.info(f'Getting results from {url}')
    result = get_results(url, filtered_params)

    output_filename = f'data/raw/dump_{int(time.time())}.json'
    logger.info(f'Exporting data to {output_filename}')
    with open(output_filename, 'w') as f:
        f.write(json.dumps(result, indent=4))

    logger.info('Creating dataset...')
    df = create_dataset(input_filepath)

    logger.info('Cleaning dataset...')
    df = clean_dataset(df)

    logger.info(f'Exporting dataset to {output_filepath}')
    export_dataset(df, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    IDEALISTA_API_KEY: str = os.environ['IDEALISTA_API_KEY']
    IDEALISTA_SECRET: str = os.environ['IDEALISTA_SECRET']

    main()
