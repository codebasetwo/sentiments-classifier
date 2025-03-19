<div align="center">
<h1> <img width="30" src="https://www.google.com/imgres?q=news%20logo&imgurl=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fpreviews%2F007%2F539%2F914%2Fnon_2x%2Fnews-logo-design-vector.jpg">&nbsp;SENTIMENTS CLASSIFIER</h1>
Classifies a Sentence if it is Neutral, Positive or Negative
</div>

<br>

<div align="center">
    <a target="_blank" href="https://www.linkedin.com/in/emekan"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/codebasetwo"><img src="https://img.shields.io/twitter/follow/codebasetwo.svg?label=Follow&style=social"></a>
</div>

<br>
<hr>

## Overview
With the increase in internet users, there is aim to regulate and moderate contents. Aiming to reduce harm and or contents that will cause conflicts there is a need for regualtion. This project aims to classify post if they are negative, positive or neutral to be able to moderate and flag contents that might or will cause harm.
<br>

This Project is an end-end machine learning project with various component:
- **Iterate:** Continously iterate on model and data
- **Scale:** Scale service as per traffic
- **CI/CD:** To continously train and and deploy better models.
- **MLOps:** Connecting software engineering best principles to machine learning workflows MLOps

## Data
My dataset was gotten from [Hugging Face](https://huggingface.co), so it was freely available. Therefore was no cost in getting data.

## Set up
All workload was carried out on my personal laptop. <br>
1. `Download` and `Install` [miniconda](https://docs.anaconda.com/miniconda/install/)


2. **Create Virtual environment**

    ```bash
    conda create -n sentiment-classifier python=3.10
    conda activate sentiment-classifier
    ```
    ```bash
    git clone https://github.com/codebasetwo/sentiments-classifier.git .
    cd sentiments-classifier
    export PYTHONPATH=$PYTHONPATH:$PWD
    pip install -r requirements.txt
    pre-commit install
    pre-commit autoupdate
    ```
3. ***Credentials***

    ```bash
    # create environment file
    touch .env
    ```
    ```bash
    # In the .env file setup any needed credentials
    GITHUB_USERNAME="CHANGE_THIS_TO_YOUR_USERNAME"  # ← CHANGE THIS
    ```

    ```bash
    source .env
    ```

## Notebook

Start by exploring the [jupyter notebook](notebooks/sentiments.ipynb) to interactively walkthrough the machine learning workflow.

```bash
  # Start notebook
  jupyter lab notebooks/sentiments.ipynb
```

## Scripts

We can also execute the same workloads in the notebooks using the clean Python scripts following software engineering best practices (training, tuning, testing, documentation, serving, versioning, etc.) `Caveat` since notebooks are mainly for iteration the code in the scripts might look more robust and better fornatted. The codes implemented in the notebook was refactored into the following scripts:

```bash
src
├──sentiments/
    ├── config.py
    ├── pro_data.py
    ├── evaluate.py
    ├── model.py
    ├── predict.py
    ├── server.py
    ├── train.py
    ├── tune.py
    └── utils.py
```


### Training
```bash
export EXPERIMENT_NAME="Sentiments-Classifier"
export DATASET_LOC="data/datasets/train.csv"
export VALIDATION_LOC="data/datasets/validation.csv"
export PARAMS='{"num_epochs": 3, "max_length": 128, "batch_size": 64, "learning_rate": 0.001}'
python src/sentiments/train.py \
    --train_dataset-loc "$DATASET_LOC" \
    --val_dataset_loc "$VALIDATION_LOC" \
    --params "$PARAMS" \
    --experiment-name "$EXPERIMENT_NAME" \

```

### Tuning
```bash
export EXPERIMENT_NAME="Sentiment-Classifier"
export DATASET_LOC="data/datasets/train.csv"
export VAL_LOC="data/datasets/validation.csv"
python src/sentiments/tuner.py \
    --dataset-loc "$DATASET_LOC" \
    --val_set_loc "$VAL_LOC"
    --experiment-name "$EXPERIMENT_NAME" \
```

### Experiment tracking

I used [MLflow](https://mlflow.org/) to track our experiments and store our models and the [MLflow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to view our experiments. [MLflow](https://mlflow.org/) helps to have a central location to store all of our experiments. It's easy and inexpensive to spin up it is also open source so can be used freely they are other managed solutions.

```bash
export MODEL_REGISTRY=$(python -c "from sentiments import config; print(config.MLFLOW_TRACKING_URI)")
mlflow server -h 127.0.0.1 -p 5000 --backend-store-uri $MODEL_REGISTRY --default-artifact-root $MODEL_REGISTRY
```
You can go to  <a href="http://localhost:8080/" target="_blank">http://localhost:5000/</a> to view your MLflow dashboard.

</details>


### Evaluation
```bash
export EXPERIMENT_NAME="Sentiment-Classifier"
export RESULTS_FILE=results/evaluation_results.json
export HOLDOUT_LOC="data/datasets/test.csv"
python src/sentiments/evaluation.py \
    --experiment_name "$EXPRIMENT_NAME" \
    --test_file_path "$HOLDOUT_LOC" \
    --results_file_path "$RESULTS_FILE" \
```
```json
{
  "overall_class_report": {
    "negative": {
      "precision": 0.7090784044016506,
      "recall": 0.666882276843467,
      "f1-score": 0.6873333333333334,
      "support": 1546.0
    },
    "neutral": {
      "precision": 0.628061224489796,
      "recall": 0.6381544841886988,
      "f1-score": 0.6330676266392389,
      "support": 1929.0
    },
    "positive": {
      "precision": 0.7520938023450586,
      "recall": 0.7786127167630058,
      "f1-score": 0.7651235444476001,
      "support": 1730.0
    },
    "accuracy": 0.693371757925072,
    "macro avg": {
      "precision": 0.6964111437455017,
      "recall": 0.6945498259317239,
      "f1-score": 0.6951748348067243,
      "support": 5205.0
    },
    "weighted avg": {
      "precision": 0.6933501620178136,
      "recall": 0.693371757925072,
      "f1-score": 0.6930775248827614,
      "support": 5205.0
    }
  },
  "slices": {
    "words_greater_than_twenty": {
      "precision": 0.6201388888888889,
      "recall": 0.6201388888888889,
      "f1": 0.6201388888888889,
      "num_samples": 1440
    },
    "words_greater_10_less_twenty": {
      "precision": 0.6974892835272505,
      "recall": 0.6974892835272505,
      "f1": 0.6974892835272505,
      "num_samples": 1633
    },
    "words_less_ten": {
      "precision": 0.7538726333907056,
      "recall": 0.7538726333907056,
      "f1": 0.7538726333907056,
      "num_samples": 1743
    }
  }
}
...

```

### Inference
```bash
export EXPERIMENT_NAME="Sentiment-Classifier"
python src/sentiments/predict.py predict \
    --experiment_name $EXPERIMENT_NAME \
    --metric "val_loss" \
    --mode "ASC" \
```

```json
[{
  "prediction":
    "neutral",
  "probabilities": {
    "negative": 0.0045743575,
    "neutral": 0.9909532,
    "positive": 0.004472421,
  }
},]
```

### Serving

  ```bash
  # Set up
  export EXPERIMENT_NAME="Sentiment-Classifier"
  python src/sentiments/serve.py --experiment_name $EXPERIMENT_NAME
  ```

  Once the application is running, we can use it via CURL, Python, etc.: <br>
  1. `Via CURL` <br>
      ```bash
      curl -X 'POST' \
            'http://127.0.0.1:8000/predict/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "data": "{\"id\": 6414,  \"text\": \"Great for organizing my work life balance\", \"label\": 2,      \"sentiment\":\"positive\"}",
            "experiment_name": "Sentiment-Classifier",
            "metric": "val_loss",
            "mode": "ASC"
            }'
      ```
  2. `Via Python` <br>
      ```python
      # Python
      import json
      import requests
      id_ = 6414
      text = "Great for organizing my work life balance"
      label = 2
      sentiment = "positive"
      data = [{"id": id_, "text": headline, "label": label,  "sentiment": sentiment}]
      requests.post("http://127.0.0.1:8000/predict", data=data).json()
      ```

### Testing
```bash
export EXPERIMENT_NAME="Sentiment-Classifier"
TEST_LOC="data/datasets/test.csv"
VAL_LOC="data/datasets/validation.csv"
DATASET_LOC="data/datasets/train.csv"
# Test data
echo "Running tests for data..."
pytest --dataset-loc=$TEST_LOC tests/data --verbose --disable-warnings | tee

# Test code
echo "Running tests for code..."
pytest --trainset-loc=$DATASET_LOC --valset-loc=$VAL_LOC tests/code --verbose --disable-warnings | tee

# Test model
echo "Running tests for models..."
pytest --experiment-name=$EXPERIMENT_NAME tests/models --verbose --disable-warnings | tee
```
