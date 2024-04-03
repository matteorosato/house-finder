# -*- coding: utf-8 -*-
import logging
import pickle
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.constants import PROCESSED_DIR, MODELS_DIR

AVAILABLE_MODELS = {
    # sample models, may be extended in the future
    'RandomForest': RandomForestRegressor(n_estimators=100, criterion='friedman_mse')
}


class ModelTrainer:

    def __init__(self, model_name: str, target_name: str = 'price'):
        """
        Initialize a ModelTrainer object.

        Args:
            model_name (str): The name of the model to be trained.
            target_name (str): The name of the target variable to be predicted.
        """
        if model_name not in AVAILABLE_MODELS.keys():
            raise ValueError(f'model_name must be one of {list(AVAILABLE_MODELS.keys())}')
        self.model_name = model_name
        self.target_name = target_name
        self.model = AVAILABLE_MODELS[self.model_name]

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    def split_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=self.target_name)
        y = df[self.target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=555)
        return X_train, X_test, y_train, y_test

    def fit_model(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def evaluate_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> \
            Dict[str, float]:
        y_pred = self.model.predict(X_test).astype(int)
        score = self.model.score(X_train, y_train)
        self.logger.info(f'R2 score on training data: {round(score, 2)}')
        cv_scores = -1 * cross_val_score(self.model,
                                         X=pd.concat([X_train, X_test], axis=0),
                                         y=pd.concat([y_train, y_test], axis=0),
                                         scoring='neg_mean_absolute_error',
                                         cv=5)
        mean_cv_score = round(cv_scores.mean(), 2)
        self.logger.info(f'Mean cross-validation score: {mean_cv_score}')
        metrics = [r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error]
        metric_results = dict()
        for metric in metrics:
            metric_results[metric.__name__] = round(metric(y_test, y_pred), 2)
        return metric_results

    def export_model(self):
        output_filename = MODELS_DIR.joinpath(self.model_name + '.pkl')
        with open(output_filename, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self, df: pd.DataFrame):
        X_train, X_val, y_train, y_val = self.split_df(df)
        self.fit_model(X_train, y_train)
        evaluation_dict = self.evaluate_model(X_train, X_val, y_train, y_val)
        self.logger.info(f"Results on validation data: {evaluation_dict}")


def main():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    my_model = ModelTrainer(model_name='RandomForest', target_name='price')
    logger.info('Loading training data...')
    training_datapath = PROCESSED_DIR.joinpath('training_data.csv')
    training_df = pd.read_csv(training_datapath, index_col='propertyCode')
    logger.info(f'Training the {my_model.model_name} model...')
    my_model.train(training_df)
    logger.info('Persisting the trained model...')
    my_model.export_model()
    logger.info('Done!')


if __name__ == '__main__':
    main()
