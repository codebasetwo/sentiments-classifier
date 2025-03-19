import json
import pickle

import pytest
from tensorflow.keras.preprocessing.text import Tokenizer

from sentiments import config, pro_data


@pytest.fixture(scope="module")
def class_to_index():
    return config.CLASS_TO_INDEX


@pytest.fixture(scope="module")
def meta_path():
    return config.METADATA_PATH


# Test clean_data method
def test_clean_data(preprocessor, df):
    # Add some NaN and duplicate rows to the DataFrame
    df.loc[3, "text"] = None
    df.loc[6] = [9235, "getting cds ready for tour", 1, "neutral"]

    cleaned_df = preprocessor.clean_data(df)

    # Check if NaN and duplicates are removed
    assert cleaned_df.isna().sum().sum() == 0
    assert len(cleaned_df) == 5


def test_tokenize(preprocessor, df, meta_path):

    preprocessor.tokenize(df)

    # Check if tokenizer and class_to_index are set
    assert preprocessor.tokenizer is not None
    assert preprocessor.class_to_index is not None

    # Verify the class_to_index file was created and contains the correct data
    class_index_file = meta_path / "class_index.json"
    assert class_index_file.exists()

    with open(class_index_file, "r") as f:
        class_to_index = json.load(f)
    assert class_to_index == {"positive": 2, "negative": 0, "neutral": 1}

    # Verify the tokenizer file was created and can be loaded
    tokenizer_file = meta_path / "tokenizer.pkl"
    assert tokenizer_file.exists()

    with open(tokenizer_file, "rb") as f:
        tokenizer = pickle.load(f)
    assert isinstance(tokenizer, Tokenizer)


def test_padding(preprocessor, df, tokenizer_class_index):
    # Set the tokenizer and class_to_index

    tokenizer, class_to_index = tokenizer_class_index
    preprocessor.tokenizer = tokenizer
    preprocessor.class_to_index = class_to_index
    train_df, y_train = preprocessor.pad_sequences(df)

    # Check if the output shapes are correct
    assert train_df.shape == (6, 128)
    assert y_train.shape == (6,)


def test_load_preprocessor():
    preprocessor = pro_data.Preprocessor.load_preprocessor()
    assert preprocessor.class_to_index == {
        "positive": 2,
        "negative": 0,
        "neutral": 1,
    }
    assert isinstance(preprocessor.tokenizer, Tokenizer)
