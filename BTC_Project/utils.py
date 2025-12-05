import os
import yaml
import pandas as pd
import pickle
from datetime import datetime
from logger import logging
from custom_exception import CustomException


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e)


def create_directories(dirs: list):
    """Create multiple directories if missing."""
    try:
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Directory ensured: {dir_path}")
    except Exception as e:
        raise CustomException(e)


def save_pickle_object(file_path: str, obj):
    """Save python object as pickle."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e)


def load_pickle_object(file_path: str):
    """Load pickle file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e)


def save_dataframe(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    try:
        df.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")
    except Exception as e:
        raise CustomException(e)


def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise CustomException(e)


def timestamp():
    """Return formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
