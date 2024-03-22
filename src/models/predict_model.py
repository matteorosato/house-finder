# -*- coding: utf-8 -*-
import logging
import os
import pickle
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR.joinpath("data/raw")  # name of the folder for raw data
PROCESSED_DIR = PROJECT_DIR.joinpath("data/processed")  # name of the folder for processed data
MODELS_DIR = PROJECT_DIR.joinpath("models")  # name of the folder for models


class Predictor:

    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        self.model_name = self.get_model_name(self.model_filepath)
        self.model = self.load_model(self.model_filepath)

    @property
    def logger(self):
        return logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    @staticmethod
    def get_model_name(filepath):
        filename = os.path.basename(filepath)
        model_name = os.path.splitext(filename)[0]
        return model_name

    @staticmethod
    def load_model(model_filepath):
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def evaluate_model(y_test, y_pred):
        metrics = [r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error]
        metric_results = dict()
        for metric in metrics:
            metric_results[metric.__name__] = round(metric(y_test, y_pred), 2)
        return metric_results

    @staticmethod
    def generate_report(cleaned_df, predictions_df) -> pd.DataFrame:
        # keep only specified columns
        columns = ['municipality', 'address', 'size', 'url']
        df = cleaned_df[columns]

        # create a new column for including price difference (predicted vs real)
        predictions_df['price_diff'] = predictions_df['predicted_price'] - predictions_df['price']

        # add info about prices removing unneeded items (inner join)
        df = df.join(predictions_df, how='inner').sort_values(by="price_diff", ascending=True)

        return df

    def predict(self, test_df) -> pd.DataFrame:
        price = test_df.pop('price')

        self.logger.info(f"Run predictions...")
        predicted_price = self.model.predict(test_df).astype(int)

        evaluation_dict = self.evaluate_model(y_test=price, y_pred=predicted_price)
        self.logger.info(f"Results on test data: {evaluation_dict}")

        predictions_df = pd.DataFrame({'price': price, 'predicted_price': predicted_price})
        return predictions_df


def main():
    model_filepath = MODELS_DIR.joinpath('RandomForest' + '.pkl')
    predictor = Predictor(model_filepath=model_filepath)

    test_datapath = PROCESSED_DIR.joinpath('test_data.csv')
    test_df = pd.read_csv(test_datapath, index_col='propertyCode')

    logger.info(f'Predicting results with {predictor.model_name} model')
    predictions_df = predictor.predict(test_df)

    cleaned_datapath = PROCESSED_DIR.joinpath('cleaned_data.csv')
    cleaned_df = pd.read_csv(cleaned_datapath, index_col='propertyCode')
    results_df = predictor.generate_report(cleaned_df, predictions_df)

    output_filepath = PROJECT_DIR.joinpath(f'results/report_{int(time.time())}.csv')
    logger.info(f'Exporting final report to to {output_filepath}')
    results_df.to_csv(output_filepath)
    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
