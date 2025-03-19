import pytest

from sentiments import predict


@pytest.mark.parametrize(
    "example_a, example_b, label",
    [
        (
            '{"id": 9425, "text": "Great app. Only complaint is that I\'d like the AMOLED dark mode in the app to match the widget.", "label":2, "sentiment": "positive"}',
            '{"id": 9425, "text": "Great app. Only complaint is I\'d like the AMOLED dark mode in the app to match the widget.", "label":2, "sentiment": "positive"}',
            "positive",
        ),
        (
            '{"id": 9425, "text": "Great for organizing my work life balance", "label":2, "sentiment": "positive"}',
            '{"id": 9425, "text": "Great for organizing my work balance", "label":2, "sentiment": "positive"}',
            "positive",
        ),
        (
            '{"id": 9425, "text": "getting cds ready for tour", "label":1, "sentiment": "neutral"}',
            '{"id": 9425, "text": "getting ready for tour", "label":1, "sentiment": "neutral"}',
            "neutral",
        ),
        (
            '{"id": 9425, "text": "Not good. Wasted time.", "label":0, "sentiment": "negative"}',
            '{"id": 9425, "text": "Not good. time wasted.", "label":0, "sentiment": "negative"}',
            "negative",
        ),
    ],
)
def test_model_invariance(example_a, example_b, label, experiment_name):
    result_a = predict.predict(example_a, experiment_name=experiment_name)
    result_b = predict.predict(example_b, experiment_name=experiment_name)

    assert result_a[0]["prediction"] == result_b[0]["prediction"] == label
