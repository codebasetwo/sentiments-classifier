# flake8: noqa: E501
import json
from datetime import datetime
from functools import partial
from typing import Dict

import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from mlflow.models.signature import infer_signature
from rich import print
from tensorflow.keras.optimizers import Adam

from sentiments import utils
from sentiments.config import SPACE, logger, mlflow
from sentiments.model import SentimentModel
from sentiments.pro_data import Preprocessor


def train_func(
    params: Dict,
    dataset_loc: str,
    val_set_loc: str,
    experiment_name: str = "Sentiment-Classifier",
    max_len: int = 128,
    trunc_type: str = "post",
    padding_type: str = "post",
    history_fp: str = None,
) -> Dict:
    """
    Trains a model using the specified parameters and dataset.

    This function trains a model (e.g., BERT-based) on the provided dataset. It supports
    optional logging of training history, custom experiment naming, and limiting the
    number of samples for training. The function returns the training results, including
    the final validation loss, training status, and the trained model.

    Args:
        params (Dict): Hyperparameters or configuration for the training process.
        dataset_loc (str): Path to the dataset used for training.
        val_set_loc (str): Path to the dataset used for training.
        experiment_name (str, optional): Name of the experiment for tracking purposes.
                                        Defaults to "Sentiment-Classifier".
        max_len (int, optional): Maximum sequence length for input data. Defaults to 128.
        trunc_type (str): how to truncate the tokenize data. Defaults to <post>
        padding_type (str): how to pad the sequence Defaults to <post>

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - "loss": The final validation loss from the training process.
            - "accuracy" The final validation accuracy from the training process
            - "status": The status of the training process (e.g., STATUS_OK).
    """

    params["max_len"] = max_len
    params["trunc_type"] = trunc_type
    params["pad_type"] = padding_type

    train_df = pd.read_csv(dataset_loc)
    val_df = pd.read_csv(val_set_loc)

    preprocessor = Preprocessor()
    train_df = preprocessor.clean_data(train_df)
    preprocessor.tokenize(train_df, vocab_size=params.get("vocab_size"))
    X_train, y_train = preprocessor.pad_sequences(
        train_df,
        max_length=max_len,
        trunc_type=trunc_type,
        padding_type=padding_type,
    )
    val_df = preprocessor.clean_data(val_df)
    X_val, y_val = preprocessor.pad_sequences(
        val_df,
        max_length=max_len,
        trunc_type=trunc_type,
        padding_type=padding_type,
    )

    # Set the experiment name. If it doesn't exist, it will be created.
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Tuning Info", "without cleaning text")

        mlflow.log_params(params)
        # # Training components
        optimizer = Adam(learning_rate=params.get("learning_rate"))

        # Instantiate the class
        sentiment_model = SentimentModel(num_epochs=params.get("num_epochs"))
        # Create model
        model = sentiment_model.create_model(
            optimizer=optimizer, vocab_size=params.get("vocab_size")
        )
        history = sentiment_model.fit_model(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=params.get("batch_size"),
        )

        metrics = utils.metrics_by_epoch(history)
        for epoch in range(params.get("num_epochs")):
            mlflow.log_metrics(metrics[epoch], step=epoch)

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        metrics["params"] = params

        if history_fp:
            utils.save_dict_to_json(metrics, path=history_fp)

        logger.info(json.dumps(metrics[0], indent=2))

        mlflow.tensorflow.log_model(
            model,
            artifact_path="sentiment_classifier",
            signature=signature,
            input_example=X_train,
        )

    return {
        "loss": history.history["val_loss"][-1],
        "accuracy": history.history["val_accuracy"][-1],
        "status": STATUS_OK,
    }


def objective(
    params,
    dataset_loc: str,
    val_set_loc: str,
    experiment_name: str = "Sentiment-Classifier",
    max_len: int = 128,
    trunc_type: str = "post",
    padding_type: str = "post",
    history_fp: str = None,
) -> Dict:
    """
    Objective function of hyperparameter.
    optimizes the hyperar=meter space.

    Args:
        params (Dict): Hyperparameters or configuration for the training process.
        dataset_loc (str): Path to the dataset used for training.
        val_set_loc (str): Path to the dataset used for training.
        experiment_name (str, optional): Name of the experiment for tracking purposes.
                                        Defaults to "Sentiment-Classifier".
        max_len (int, optional): Maximum sequence length for input data. Defaults to 128.
        trunc_type (str, optional): how to truncate the tokenize data. Defaults to <post>
        padding_type (str, optional): how to pad the sequence Defaults to <post>

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - "loss": The final validation loss from the training process.
            - "accuracy" The final validation accuracy from the training process
            - "status": The status of the training process (e.g., STATUS_OK).
    """

    print(f"Evaluating hyperparameters: {params}")

    result = train_func(
        params=params,
        dataset_loc=dataset_loc,
        val_set_loc=val_set_loc,
        experiment_name=experiment_name,
        max_len=max_len,
        trunc_type=trunc_type,
        padding_type=padding_type,
        history_fp=history_fp,
    )

    return result  # Return the validation loss for Hyperopt


# Hyperopt optimization
def tune_hyperparameters(
    dataset_loc: str,
    val_set_loc: str,
    experiment_name: str = "Sentiment-Classifier",
    max_len: int = 128,
    trunc_type: str = "post",
    padding_type: str = "post",
    max_evals: int = 5,
    history_fp: str = None,
) -> Dict:
    """_summary_

    Args:
        dataset_loc (str): Path to the dataset used for training
        val_set_loc (str): Path to the dataset used for validation
        experiment_name (str, optional): name of experiment. Defaults to "Sentiment-Classifier".
        max_len (int, optional): maximum length of token. Defaults to 128.
        trunc_type (str, optional): truncation type. Defaults to "post".
        padding_type (str, optional): padding type. Defaults to "post".
        max_evals (int, optional): maximum number of times to run search. Defaults to 5.
        history_fp (str, optional): path to store searches. Defaults to None.

    Returns:
        Dict: best parameters of tune model.
    """

    trials = Trials()
    wrapped_objective = partial(
        objective,
        dataset_loc=dataset_loc,
        val_set_loc=val_set_loc,
        experiment_name=experiment_name,
        max_len=max_len,
        trunc_type=trunc_type,
        padding_type=padding_type,
        history_fp=history_fp,
    )

    best = fmin(
        fn=wrapped_objective,
        space=SPACE,  # search space
        algo=tpe.suggest,
        max_evals=max_evals,  # Number of hyperparameter combinations to try
        trials=trials,
    )

    print(":white_check_mark:", "Search Done.")
    if history_fp:
        with open(history_fp, "rb") as file:
            data = json.load(file)

    best_params = utils.get_best_params(data)

    logger.info(json.dumps(data))

    return best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="tuner.py",
        description="Tune hyperparameters for the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_loc", type=str, help="Path to the train_dataset."
    )

    parser.add_argument(
        "val_set_loc", type=str, help="Path to the validation dataset."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Sentiment-Classifier",
        help="Name of the experiment.",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length of tokenization.",
    )

    parser.add_argument(
        "--trunc_type",
        type=str,
        default="post",
        choices=["post", "pre"],
        help="how to truncate sequences length post after and pre before",
    )
    parser.add_argument(
        "--pad_type",
        type=str,
        default="post",
        choices=["post", "pre"],
        help="where to pad sequences post after and pre before",
    )

    parser.add_argument(
        "--max_evals",
        type=int,
        help="number of times to run hyper parameter serach",
        default=5,
    )
    parser.add_argument(
        "--history_fp",
        type=str,
        default=None,
        help="Path to save the training history.",
    )
    args = parser.parse_args()

    best_params = tune_hyperparameters(
        dataset_loc=args.dataset_loc,
        val_set_loc=args.val_set_loc,
        experiment_name=args.experiment_name,
        max_len=args.max_len,
        trunc_type=args.trunc_type,
        padding_type=args.pad_type,
        max_evals=args.max_evals,
        history_fp=args.history_fp,
    )
    print("Best hyperparameters:", best_params)
