import pytest


def pytest_addoption(parser):
    """Add option to specify name of experiment when executing tests from CLI.
    Ex: pytest --experiment-name=$EXPERIMENT_NAME tests/model --verbose --disable-warnings
    """
    parser.addoption(
        "--experiment-name",
        action="store",
        default="Sentiment-Classifier",
        help="name of your experiment",
    )


@pytest.fixture(scope="module")
def experiment_name(request):
    """Load dataset as a Great Expectations object."""
    experiment_name = request.config.getoption("--experiment-name")
    return experiment_name
