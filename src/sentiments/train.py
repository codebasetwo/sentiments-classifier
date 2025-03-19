import json
from datetime import datetime

import pandas as pd
import tensorflow as tf
from mlflow.models import infer_signature
from rich import print
from tensorflow.keras.optimizers import Adam

from sentiments import utils
from sentiments.config import logger, mlflow
from sentiments.model import SentimentModel
from sentiments.pro_data import Preprocessor


def trainer(
    train_dataset_loc: str,
    val_dataset_loc: str,
    params: str = None,
    experiment_name: str = "Sentiment-Classifier",
    num_epochs: int = 3,
    batch_size: int = 32,
    max_len: int = 128,
    learning_rate: float = 0.001,
    vocab_size: int = 5000,
    trunc_type: str = "post",
    padding_type: str = "post",
    history_fp: str = None,
) -> tf.keras.callbacks.History:
    """
    Train model on datasets.

    Args:
        train_dataset_loc (str): path to training data.
        val_dataset_loc (str): path to validation data.
        params (str): A string of Dictionary of parameters to pass to model Defaults None if passed ignores passed parameter
        experiment_name (dtr): name of experiment. Defaults "Sentiment-Classifer"
        num_epochs (int): number of epochs. Defaults (3)
        batch_size (int): number batch Defaults 32.
        max_len (int): maximum lenth of tokenize sequence Defaults (128)
        learning_rate (float): learning rate of model for gradient desecent Defaults (0.001)
        vocab_size (int): size of vocabulary from training data. Defaults (5000)
        trunc_type (str): how to truncate seunece one of <pre> <post> defaults (post)
        padding_type (str): type of padding one of <pre> <post> defaults (post)
        history_fp (str): path to store training history if None does not store trainig history Defaults (None)

        Returns:
            history (tf.keras.callbacks.History): history of the training run
    """
    if params:
        params = json.loads(params)
    else:
        params = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "trunc_type": trunc_type,
            "padding_type": padding_type,
            "max_len": max_len,
        }
    train_df = pd.read_csv(train_dataset_loc)
    val_df = pd.read_csv(val_dataset_loc)

    preprocessor = Preprocessor()
    train_df = preprocessor.clean_data(train_df)
    preprocessor.tokenize(train_df)
    X_train, y_train = preprocessor.pad_sequences(
        train_df,
        max_length=max_len,
        trunc_type=params.get("trunc_type") or trunc_type,
        padding_type=params.get("padding_type") or padding_type,
    )
    val_df = preprocessor.clean_data(val_df)
    X_val, y_val = preprocessor.pad_sequences(val_df)

    # Training components
    optimizer = Adam(
        learning_rate=params.get("learning_rate") or learning_rate
    )

    # Instantiate the class
    sentiment_model = SentimentModel(
        num_epochs=(params.get("num_epochs") or num_epochs)
    )
    # Create model
    model = sentiment_model.create_model(
        optimizer=optimizer,
        vocab_size=(params.get("vocab_size", None) or vocab_size),
    )
    history = sentiment_model.fit_model(
        model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=params.get("batch_size") or batch_size,
    )

    metrics = {
        "val_loss": history.history["val_loss"][-1],
        "val_accuracy": history.history["val_accuracy"][-1],
    }

    # Create MLflow Experiment
    mlflow.set_experiment(experiment_name)
    # Start an MLflow run
    with mlflow.start_run() as run:

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metrics(metrics)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "without cleaning text")

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        model_info = mlflow.tensorflow.log_model(
            model,
            artifact_path="sentiment_classifier",
            signature=signature,
            input_example=X_train,
        )

    # Save the run data
    data = {
        "timestamp": datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "model_artifact_path": model_info.artifact_path,
        "run_id": run.info.run_id,
        "params": params,
        "metrics": utils.metrics_by_epoch(history),
    }

    print(":white_check_mark:", "Training Done.")
    logger.info(json.dumps(data, indent=2))
    if history_fp:
        utils.save_dict_to_json(data, history_fp)

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a text classification news_model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("train_loc", type=str, help="Path to the dataset")
    parser.add_argument(
        "val_loc", type=str, help="Path to the validation dataset"
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Hyperparameters dictionary for the model if passed ignore passing individual hyperparameters",
        default=None,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default="Sentiment-Classifier",
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of epochs to train", default=3
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training", default=32
    )
    parser.add_argument(
        "--max_len",
        type=int,
        help="Maximum length of the input token sequence",
        default=128,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate for the optimizer",
        default=0.001,
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        help="size of vocabulary in data to use.",
        default=5000,
    )

    parser.add_argument(
        "--trunc",
        type=str,
        help="how to truncate the length of sequence",
        default="post",
        choices=["post", "pre"],
    )

    parser.add_argument(
        "--pad",
        type=str,
        help="where to pad the sequences with zeros",
        default="post",
        choices=["post", "pre"],
    )

    parser.add_argument(
        "--history_fp",
        type=str,
        help="Path to save the training history",
        default=None,
    )

    args = parser.parse_args()
    trainer(
        args.train_loc,
        args.val_loc,
        params=args.params,
        experiment_name=args.experiment_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        learning_rate=args.learning_rate,
        vocab_size=args.vocab_size,
        trunc_type=args.trunc,
        padding_type=args.pad,
        history_fp=args.history_fp,
    )
