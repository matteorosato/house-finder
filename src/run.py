# -*- coding: utf-8 -*-
from src.data import make_dataset
from src.models import train_model, predict_model


def main():
    """
    Wrapper for running the main functions of the tool. (Un)comment one or more step if (un)needed.
    """
    # 1. Download the data and create the dataset
    make_dataset.main()
    # 2. Train the model
    train_model.main()
    # 3. Test the model -> get the report!
    predict_model.main()


if __name__ == "__main__":
    main()
