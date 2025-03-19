# flake8: noqa:  F841
import kfp.compiler
from kfp import dsl, local

local.init(runner=local.DockerRunner())


@dsl.component(base_image="mekuzeeyo/sentiments:0.0.1")
def download_dataset(
    repo_name: str,
    train_set: dsl.Output[dsl.Dataset],
    val_set: dsl.Output[dsl.Dataset],
    test_set: dsl.Output[dsl.Dataset],
):
    from datasets import load_dataset

    dataset = load_dataset(repo_name)
    dataset["train"].to_csv(
        train_set.path,
        index=False,
    )
    dataset["test"].to_csv(val_set.path, index=False)
    dataset["validation"].to_csv(test_set.path, index=False)


@dsl.component(base_image="mekuzeeyo/sentiments:0.0.1")
def train(
    training_data: dsl.Input[dsl.Dataset],
    val_data: dsl.Input[dsl.Dataset],
    training_history: dsl.Output[dsl.Artifact],
):
    import json

    from sentiments import train

    history = train.trainer(
        train_dataset_loc=training_data.path,
        val_dataset_loc=val_data.path,
        batch_size=512,
        num_epochs=1,
    )

    with open(training_history.path, "w") as fp:
        json.dump(history.history, fp=fp)


@dsl.component(base_image="mekuzeeyo/sentiments:0.0.1")
def tune(
    max_eval: int,
    training_data: dsl.Input[dsl.Dataset],
    val_data: dsl.Input[dsl.Dataset],
    tuning_history: dsl.Output[dsl.Artifact],
):
    import json

    from sentiments import tuner

    best_params = tuner.tune_hyperparameters(
        train_dataset_loc=training_data.path,
        val_dataset_loc=val_data.path,
        max_evals=max_eval,
    )

    with open(tuning_history.path, "w") as fp:
        json.dump(best_params, fp=fp)


@dsl.component(
    base_image="mekuzeeyo/sentiments:0.0.1", packages_to_install=["snorkel"]
)
def evaluate(
    test_data: dsl.Input[dsl.Dataset],
    result_path: dsl.Output[dsl.Artifact],
):
    import json

    from sentiments import evaluation

    metrics = evaluation.evaluate(file_path=test_data.path)
    with open(result_path.path, "w") as fp:
        json.dump(metrics, fp=fp)


@dsl.pipeline
def sentiment_pipeline(repo_name: str, max_eval: int):
    download_task = download_dataset(repo_name=repo_name)
    training_task = train(
        training_data=download_task.outputs["train_set"],
        val_data=download_task.outputs["val_set"],
    )  # noqa: F401
    tuning_task = tune(
        max_eval=max_eval,
        training_data=download_task.outputs["train_set"],
        val_data=download_task.outputs["val_set"],
    )  # noqa: F401

    evaluation_task = evaluate(
        test_data=download_task.outputs["test_set"]
    )  # noqa: F401


if __name__ == "__main__":
    repo_name = "Sp1786/multiclass-sentiment-analysis-dataset"
    max_eval = 3
    sentiment_pipeline(repo_name=repo_name, max_eval=max_eval)

    import kfp

    kfp.compiler.Compiler().compile(
        sentiment_pipeline,
        package_path="./sentiment.yaml",
        pipeline_name="sentiments",
    )
