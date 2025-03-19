import pytest
import utilities

from sentiments import train


@pytest.mark.training("compute intensive worload")
@pytest.mark.parametrize("num_epochs, expected", [(1, 1), (2, 2), (3, 3)])
def test_training_func(train_dataset, val_dataset, num_epochs, expected):
    experiment_name = utilities.generate_experiment_name()

    history = train.trainer(
        train_dataset_loc=train_dataset,
        val_dataset_loc=val_dataset,
        experiment_name=experiment_name,
        num_epochs=num_epochs,
        batch_size=512,
        learning_rate=0.1,
    )

    utilities.delete_experiment(experiment_name=experiment_name)
    assert len(history.history["val_loss"]) == expected
