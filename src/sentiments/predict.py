import json
from typing import Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from rich import print

from sentiments import pro_data
from sentiments.config import logger


def get_best_run_id(
    experiment_name: Optional[str] = "Sentiment-Classifier",
    metric: str = "val_loss",
    mode: str = "ASC",
) -> str:
    """Retrieves the run ID of the best performing run for a given experiment and metric.

    Searches MLflow runs for the specified experiment, orders them by the given metric
    (either ascending or descending), and returns the run ID of the top-performing run.

    Args:
        experiment_name: The name of the MLflow experiment. If None, searches across all experiments.
        metric: The name of the metric to use for determining the best run (e.g., "accuracy", "loss").
        mode: The sorting mode for the metric ("ASC" for ascending, "DESC" for descending).

    Returns:
        The run ID of the best run.

        Raise:
                ValueError: if `mode` is not "ASC" or "DESC"
                IndexError: If the search results are empty (no runs found).  This is handled specifically because it's a likely scenario.
    """

    if mode not in ("ASC", "DESC"):
        raise ValueError("mode must be either 'ASC' or 'DESC'.")

    if experiment_name is None:
        runs = mlflow.search_runs(order_by=[f"metrics.{metric} {mode}"])
    else:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=[f"metrics.{metric} {mode}"],
        )

    if runs.empty:
        raise IndexError(
            f"No runs found for experiment '{experiment_name or 'all'}' with metric '{metric}'."
        )

    if f"metrics.{metric}" not in runs.columns:
        raise ValueError(f"Metric '{metric}' not found in experiment runs.")

        # Get the best run
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_metric = best_run[f"metrics.{metric}"]

    print(
        f"Retrieved best {metric}: [green]{best_metric}[green\\] \nRun ID: [magenta]{best_run_id}[magenta\\]"
    )

    return best_run_id


def get_best_model(
    run_id: str, arttifact_path: str = "sentiment_classifier"
) -> tf.keras.Model:
    """Loads the best trained Keras model from a specified MLflow run.

    Args:
        run_id: The ID of the MLflow run containing the desired model.
        artifact_path: The path to mlflow artifact. Default to sentiment_classifier

    Returns:
        The loaded Keras model.
    """
    # Load the best model
    try:
        model = mlflow.tensorflow.load_model(
            f"runs:/{run_id}/{arttifact_path}"
        )
    except Exception as e:
        logger.error("An error occured %s", f"{e}")
    return model


def format_probability(
    probabilities: np.ndarray, index_to_class: Dict[int, str]
) -> Dict[str, float]:
    """
    Formats raw class probabilities into a dictionary mapping class labels to their probabilities.

    This function takes an array of probabilities (e.g., from a softmax output) and a dictionary
    mapping class indices to their corresponding labels. It returns a dictionary where each key
    is a class label, and the value is the probability assigned to that class.

    Args:
        probabilities (np.ndarray): A 1D array of probabilities for each class.
        index_to_class (Dict[int, str]): A dictionary mapping class indices to their corresponding labels.

    Returns:
        Dict[str, float]: A dictionary where keys are class labels and values are their probabilities.

        Example:
        >>> probabilities = np.array([0.1, 0.7, 0.2])
        >>> index_to_class = {0: "cat", 1: "dog", 2: "bird"}
        >>> format_probability(probabilities, index_to_class)
        {"cat": 0.1, "dog": 0.7, "bird": 0.2}

    """
    all_prob = {}
    for i, item in enumerate(probabilities):
        all_prob[index_to_class[i]] = item
    return all_prob


def predict_proba(test_df: pd.DataFrame, model: tf.keras.Model) -> Dict:
    """
    Generates class probabilities and predictions for a given test dataset using a trained model.

    This function preprocesses the test dataset, computes logits using the provided model,
    and converts the logits into class probabilities using softmax. It then maps the predicted
    class indices to their corresponding labels and returns the results in a structured format.

    Args:
        test_df (pd.DataFrame): The test dataset to make predictions on.
        model (tf.keras.Model): A trained TensorFlow/Keras model used for inference.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
            - "prediction": The predicted class label.
            - "probabilities": A dictionary mapping class labels to their corresponding probabilities.
    """

    preprocessor = pro_data.Preprocessor.load_preprocessor()
    class_to_index = preprocessor.class_to_index
    index_to_class = {idx: cls for cls, idx in class_to_index.items()}
    test_df = preprocessor.clean_data(test_df)
    test_df, _ = preprocessor.pad_sequences(test_df)

    # Prediction(s)
    y_pred = model.predict(test_df)

    results = []  # Empty list to store result
    for pred in y_pred.tolist():

        indices = np.argmax(pred, axis=-1)
        category = [index_to_class[index] for index in [indices]][0]
        results.append(
            {
                "prediction": category,
                "probabilities": format_probability(pred, index_to_class),
            }
        )

    return results


def predict(
    data: str,
    experiment_name: str = "Sentiment-Classifier",
    metric: str = "val_loss",
    mode: str = "ASC",
) -> List[Dict[str, Union[str, Dict[str, float]]]]:
    """Generates predictions using the best model from a specified MLflow experiment.

    Loads the best performing model (based on the given metric and mode) from the specified MLflow
    experiment, preprocesses the input dataset, generates predictions, and formats the predictions
    into a list of dictionaries.

    Args:
        data (str): the data for prediction
        experiment_name: The name of the MLflow experiment.
        metric: The metric used to determine the best model (e.g., "accuracy", "loss").
        mode: The sorting mode for the metric ("ASC" for ascending, "DESC" for descending).

    Returns:
        A list of dictionaries. Each dictionary represents a prediction for a single data point
        and contains the following keys:
            - "prediction": The predicted class label (string).
            - "probabilities": A dictionary containing class names as keys and their corresponding
              probabilities (floats) as values.
    """
    data = json.loads(data)
    data = pd.DataFrame([data])
    # Get best model run_id
    run_id = get_best_run_id(
        experiment_name=experiment_name, metric=metric, mode=mode
    )

    model = get_best_model(run_id)
    # Load the best model
    results = predict_proba(data, model)

    logger.info(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="make inference using best model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "data",
        type=str,
        help="str of sample data for inference.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="name of experiment to search",
        default="Sentiment-Classifier",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="metric to use for the search of best preferred model",
        default="val_loss",
        choices=["val_loss", "val_accuracy", "accuracy", "loss"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="how to sort the metric to return",
        default="ASC",
        choices=["ASC", "DESC"],
    )

    args = parser.parse_args()
    results = predict(
        args.data,
        experiment_name=args.experiment_name,
        metric=args.metric,
        mode=args.mode,
    )
