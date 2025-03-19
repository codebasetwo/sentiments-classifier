import json
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from sentiments.config import CLASS_TO_INDEX, DATASET_PATH


def download_data(
    hugging_face_repo_name: str = "Sp1786/multiclass-sentiment-analysis-dataset",
    directory: str = DATASET_PATH,
):
    """Download data from hugging face

    Args:
        hugging_face_repo_name (str): the name of the hugging face repo to download data from. defaults (Sp1786/multiclass-dataset-analysis-dataset)
        directory (str): path to store the downloaded data."""

    dataset = load_dataset(hugging_face_repo_name)

    dataset["train"].to_csv(f"{directory}/train.csv", index=False)
    dataset["test"].to_csv(f"{directory}/test.csv", index=False)
    dataset["validation"].to_csv(f"{directory}/validation.csv", index=False)


# Index to label
def decode(
    indices: List[int], class_index: Dict[int, str] = CLASS_TO_INDEX
) -> List[str]:
    """
    Converts a list of integer indices back to their corresponding string labels.

    This function takes a list of integer indices and a mapping dictionary (index-to-label)
    and returns a list of string labels corresponding to the provided indices.

    Args:
        indices (List[int]): A list of integer indices to be converted to string labels.
        class_index (Dict[int, str]): A dictionary mapping string labels to their
                                         corresponding integer indices .

    Returns:
        List[str]: A list of string labels corresponding to the input indices.

    Example:
        >>> index_to_class = {0: 'apple', 1: 'banana', 2: 'orange'}
        >>> decode([0, 1, 2], index_to_class)
        ['apple', 'banana', 'orange']
    """
    index_to_class = {v: k for k, v in class_index.items()}
    return [index_to_class[index] for index in indices]


def metrics_by_epoch(history: Dict) -> Dict:
    """Get the metrics by epoch.

    Args:
        history (Dict): history of the model.

    Returns:
        Dict: metrics by epoch.
    """
    metrics = {}
    for epoch, (loss, accuracy, val_loss, val_accuracy) in enumerate(
        zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        )
    ):
        metrics[epoch] = {
            "train_loss": loss,
            "train_accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    return metrics


def save_dict_to_json(
    data: Dict, path: str, cls: Optional[Any] = None, sortkeys: bool = False
) -> None:
    """Save a dictionary to a specific location.

    Args:
        data (Dict): data to save.
        path (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):  # pragma: no cover
        os.makedirs(directory)

    # Check if the file exists
    if not os.path.exists(path):
        # Writing dictionary to a new JSON file
        with open(path, "w") as fp:
            json.dump([data], fp=fp, cls=cls, sort_keys=sortkeys)
            fp.write("\n")
    else:
        # Reading the existing data from the JSON file
        with open(path, "r") as fp:
            json_file = json.load(fp)

        # Appending new data to the list
        json_file.append(data)

        # Writing the updated list back to the JSON file
        with open(path, "w") as fp:
            json.dump(json_file, fp)
            fp.write("\n")
    print(f"Data saved to {path}")


def get_best_params(data: List[Dict]) -> Dict:
    """
    Finds the best set of parameters from a list of experiment results based on the lowest validation loss.

    Args:
        data: A list of dictionaries, where each dictionary represents an experiment.
              Each experiment dictionary should contain:
                - "params": A dictionary of the parameters used in the experiment.
                - Epoch-based keys (e.g., "1", "10", "50"): Dictionaries containing metrics for each epoch, including "val_loss".

    Returns:
        A dictionary containing the "params" of the experiment with the lowest validation loss in the last epoch.
        Returns None if the input data is empty or if no experiment contains valid epoch data.

    Example:
        experiments = [
            {
                "params": {"lr": 0.01, "batch_size": 32},
                "1": {"val_loss": 0.5},
                "10": {"val_loss": 0.3},
                "20": {"val_loss": 0.25}
            },
            {
                "params": {"lr": 0.001, "batch_size": 64},
                "1": {"val_loss": 0.6},
                "10": {"val_loss": 0.4},
                "30": {"val_loss": 0.2}
            }
        ]

        best_params = get_best_params(experiments)
        print(best_params)  # Output: {'lr': 0.001, 'batch_size': 64}
    """

    # Initialize variables to track the best parameters
    best_val_loss = float("inf")  # Start with a very high value
    best_params = None

    # Iterate through each experiment
    for experiment in data:
        # Get the last epoch key
        last_epoch_key = str(
            max(int(k) for k in experiment.keys() if k.isdigit())
        )

        # Get the val_loss of the last epoch
        last_epoch_val_loss = experiment[last_epoch_key]["val_loss"]

        # Check if this is the best val_loss so far
        if last_epoch_val_loss < best_val_loss:
            best_val_loss = last_epoch_val_loss
            best_params = experiment["params"]

    return best_params
