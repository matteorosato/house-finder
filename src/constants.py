"""This module defines project-level constants"""
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR.joinpath("data/raw")  # name of the folder for raw data
PROCESSED_DIR = PROJECT_DIR.joinpath("data/processed")  # name of the folder for processed data
MODELS_DIR = PROJECT_DIR.joinpath("models")  # name of the folder for models
RESULTS_DIR = PROJECT_DIR.joinpath("results")  # name of the folder for storing results
