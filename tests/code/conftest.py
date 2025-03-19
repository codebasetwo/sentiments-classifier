import json
import pickle

import pandas as pd
import pytest

from sentiments import config, pro_data


def pytest_addoption(parser):
    """Add options to specify dataset locations when executing tests from CLI.

    Ex: pytest --trainset-loc=$DATASET_LOC --valset-loc=$DATASET_LOC tests/code --verbose --disable-warnings
    """
    parser.addoption(
        "--trainset-loc",
        action="store",
        default=None,
        help="train data location.",
    )
    parser.addoption(
        "--valset-loc",
        action="store",
        default=None,
        help="validation data location.",
    )


@pytest.fixture(scope="module")
def train_dataset(request):
    """Load training dataset path."""
    train_dataset_loc = request.config.getoption("--trainset-loc")

    return train_dataset_loc


@pytest.fixture(scope="module")
def val_dataset(request):
    """Load validation dataset path."""
    val_dataset_loc = request.config.getoption("--valset-loc")
    return val_dataset_loc


@pytest.fixture(scope="module")
def preprocessor():
    return pro_data.Preprocessor()


@pytest.fixture(scope="function")
def df():
    MOCK_DATA = {
        "id": [9235, 16790, 24840, 20744, 6414, 22370],
        "text": [
            "getting cds ready for tour",
            " MC, happy mother`s day to your mom ;).. love yah",
            "A year from now is graduation....i am pretty sure i`m not ready for it!?!?!?",
            "because you had chips and sale w/o me",
            "Great for organising my work life balance",
            "its my going away partyyy  `s.  you should come!",
        ],
        "label": [1, 2, 0, 1, 2, 1],
        "sentiment": [
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "neutral",
        ],
    }
    data = pd.DataFrame(MOCK_DATA)
    return data


@pytest.fixture(scope="function")
def tokenizer_class_index():
    # Load the tokenizer to file
    with open(config.METADATA_PATH / "tokenizer.pkl", "rb") as fp:
        tokenizer = pickle.load(fp)

    # Save the class index to file
    with open(config.METADATA_PATH / "class_index.json", "rb") as fp:
        class_to_index = json.load(fp)

    return tokenizer, class_to_index
