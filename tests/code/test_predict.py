import numpy as np
import pytest

from sentiments import predict


@pytest.fixture(scope="module")
def probabilities():
    return np.array([0.1, 0.7, 0.2])


@pytest.fixture(scope="module")
def index_to_class():
    return {0: "negative", 1: "neutral", 2: "positive"}


def test_format_probability(probabilities, index_to_class):
    value = predict.format_probability(probabilities, index_to_class)

    assert value == {"negative": 0.1, "neutral": 0.7, "positive": 0.2}
