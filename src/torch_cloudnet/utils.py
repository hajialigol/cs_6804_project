from typing import Dict
from pathlib import Path
import pickle


def save_predictions(predictions_dict: Dict, write_path: Path):
    """
    Method to write predictions as inputs
    :param predictions_dict:
    :param write_path:
    :return:
    """
    with open(write_path, 'wb') as fp:
        pickle.dump(predictions_dict, fp)


def read_predictions(read_path: Path):
    """

    :param read_path:
    :return:
    """
    with open(read_path, 'rb') as fp:
        return pickle.load(fp)
