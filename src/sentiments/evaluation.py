import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)
from snorkel.slicing import PandasSFApplier, slicing_function

from sentiments.config import logger
from sentiments.predict import get_best_model, get_best_run_id
from sentiments.pro_data import Preprocessor
from sentiments.utils import decode, save_dict_to_json


@slicing_function()
def words_less_ten(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 10  # less than 10 words


@slicing_function()
def words_greater_10_less_twenty(x):
    """Projects with words less greater than 10 but less than twenty."""
    return (len(x.text.split()) > 10) and (
        len(x.text.split()) < 20
    )  # less than 20 words, greater than 10


@slicing_function()
def words_greater_than_twenty(x):
    """Projects with longer words ie above 20."""
    return len(x.text.split()) > 20  # greater than 20 words


def get_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame
) -> Dict:
    """Get performance metrics for slices.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        df (Dataset): pandas dataframe dataset with labels.
    Returns:
        Dict: performance metrics for slices.
    """
    slice_metrics = {}
    slicing_functions = [
        words_greater_than_twenty,
        words_greater_10_less_twenty,
        words_less_ten,
    ]
    applier = PandasSFApplier(slicing_functions)
    slices = applier.apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)  # create mask
        if sum(mask):  # Check atleast one sample in the slice
            metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            slice_metrics[slice_name] = {}
            slice_metrics[slice_name]["precision"] = metrics[0]
            slice_metrics[slice_name]["recall"] = metrics[1]
            slice_metrics[slice_name]["f1"] = metrics[2]
            slice_metrics[slice_name]["num_samples"] = len(y_true[mask])

    return slice_metrics


def get_class_and_aggregate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_to_index: Dict
) -> Dict:
    """
    Computes and aggregates evaluation metrics for classification performance.

    This function calculates precision, recall, F1-score (weighted average), and accuracy
    based on the true labels (`y_true`) and predicted labels (`y_pred`). It returns these
    metrics as a dictionary.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values as returned by a classifier.
        class_to_index (dictionary): dictionary mapping integer to string labels

    Returns:
        Dict[str, float]: A dictionary containing the following metrics:
            - "precision": Weighted average precision score.
            - "recall": Weighted average recall score.
            - "f1_score": Weighted average F1-score.
            - "accuracy": Accuracy score.

    Example:
        >>> y_true = [0, 1, 2, 0, 1]
        >>> y_pred = [0, 2, 1, 0, 0]
        >>> aggregate_metrics(y_true, y_pred)
        {'precision': 0.3, 'recall': 0.4, 'f1_score': 0.33, 'accuracy': 0.4}
    """

    indices = list(class_to_index.values())
    indices.sort()
    labels = decode(indices=indices, class_index=class_to_index)

    class_report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    return class_report


def evaluate(
    file_path: str,
    experiment_name: str = "Sentiment-Classifier",
    metric: str = "val_loss",
    mode: str = "ASC",
    num_samples: Optional[int] = None,
    result_path: Optional[str] = None,
):

    df = pd.read_csv(Path(file_path).absolute(), nrows=num_samples)
    preprocessor = Preprocessor.load_preprocessor()
    df = preprocessor.clean_data(df)
    X_test, y_true = preprocessor.pad_sequences(df)

    run_id = get_best_run_id(
        experiment_name=experiment_name, metric=metric, mode=mode
    )

    model = get_best_model(run_id=run_id)
    print(X_test)
    probabilities = model.predict(X_test)
    y_pred = np.argmax(probabilities, axis=-1)

    class_report = get_class_and_aggregate_metrics(
        y_true, y_pred, preprocessor.class_to_index
    )
    slice_metrics = get_slice_metrics(y_true, y_pred, df)

    metrics = {"overall_class_report": class_report, "slices": slice_metrics}

    logger.info(json.dumps(metrics, indent=2))
    if result_path:
        save_dict_to_json(metrics, result_path)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate your model, on your test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="File path to test dataset with labels to evaluate on",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default="Sentiment-Classifier",
    )

    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to evaluate the model",
        default="val_loss",
        choices=["val_loss", "val_accuracy", "accuracy", "loss"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to evaluate the model",
        default="ASC",
        choices=["ASC", "DESC"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to use for evaluation",
        default=None,
    )

    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to save the evaluation results",
        default=None,
    )

    args = parser.parse_args()

    metrics = evaluate(
        file_path=args.file_path,
        experiment_name=args.experiment_name,
        metric=args.metric,
        mode=args.mode,
        num_samples=args.num_samples,
        result_path=args.results_path,
    )
