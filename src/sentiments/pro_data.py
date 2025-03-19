import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sentiments.config import METADATA_PATH


class Preprocessor:
    def __init__(self, tokenizer=None, class_to_index=None):
        self.tokenizer = tokenizer
        self.class_to_index = class_to_index

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs cleaning on your data (removes None rows and drop duplicates)

        Args:
            df (pd.DataFrame): dataset to be cleaned.

        Returns:
            pd.DataFrame: A pandas Data of the cleaned data
        """

        # drop null or empty rows
        df = df.dropna(subset=["text", "sentiment"], ignore_index=True)
        # drop duplicate row
        df = df.drop_duplicates(
            subset=["text", "sentiment"], ignore_index=True
        )  # data leakage
        return df

    def tokenize(
        self,
        df: pd.DataFrame,
        vocab_size: int = 5000,
        oov_token: str = "<OOV>",
    ):
        """Create tokens from your data.

        Args:
            df (pd.DataFrame): data to be used for tokeninzation
            vocab_size (int, optional): the size of vocabulary to be used. Defaults to 5000.
            oov_token (str, optional): value to use to represent out of vocabulary tokens. Defaults to "<OOV>".
        """

        labels = df["label"].to_list()
        sentiments = df["sentiment"].to_list()
        # Create the mapping dictionary
        class_to_index = dict(zip(sentiments, labels))

        # Save the class index to file
        with open(Path(METADATA_PATH, "class_index.json"), "w") as f:
            json.dump(class_to_index, f)

        # Initialize the Tokenizer class
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

        # Generate the word index dictionary
        tokenizer.fit_on_texts(df["text"])

        # Save the tokenizer to file
        with open(Path(METADATA_PATH, "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)

        self.class_to_index = class_to_index
        self.tokenizer = tokenizer

    def pad_sequences(
        self,
        df: pd.DataFrame,
        max_length=128,
        trunc_type="post",
        padding_type="post",
    ):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            max_length (int, optional): _description_. Defaults to 128.
            trunc_type (str, optional): _description_. Defaults to "post".
            padding_type (str, optional): _description_. Defaults to "post".

        Returns:
            _type_: _description_
        """
        # Generate and pad the training sequences
        train_sequences = self.tokenizer.texts_to_sequences(df["text"])
        train_df = pad_sequences(
            train_sequences,
            maxlen=max_length,
            padding=padding_type,
            truncating=trunc_type,
        )
        y_train = np.array(df["sentiment"].map(self.class_to_index))

        return train_df, y_train

    @classmethod
    def load_preprocessor(cls) -> "Preprocessor":
        """Creates an instance of the Preprocessir class

        Returns:
            Preprocessor: an instance of the class.
        """
        # Load the tokenizer to file
        with open(Path(METADATA_PATH, "tokenizer.pkl"), "rb") as fp:
            tokenizer = pickle.load(fp)

        # Save the class index to file
        with open(Path(METADATA_PATH, "class_index.json"), "rb") as fp:
            class_to_index = json.load(fp)

        return cls(tokenizer=tokenizer, class_to_index=class_to_index)


if __name__ == "__main__":
    from config import DATASET_PATH

    df = pd.read_csv(Path(DATASET_PATH, "train.csv"))
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)
    preprocessor.tokenize(df)
    X_train, y_train = preprocessor.pad_sequences(df)

    print(X_train)
    print(y_train)
