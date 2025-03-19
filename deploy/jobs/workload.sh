#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir results

export TRAIN_LOC="data/datasets/train.csv"
export VAL_LOC="data/datasets/validation.csv"
export TEST_LOC="data/datasets/test.csv"
export EXPERIMENT_NAME="Sentiment-Classifier"

python src/sentiments/train.py $TRAIN_LOC $VAL_LOC --experiment-name=$EXPERIMENT_NAME   # train model
python src/sentiments/evaluation.py $TEST_LOC                                           # evaluate model
python src/sentiments/server.py                                                         # serve model

# test training data, validation data, test data
# Test data
export RESULTS_FILE=results/test_train_data_results.txt
pytest --dataset-loc=$TRAIN_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE

export RESULTS_FILE=results/test_val_data_results.txt
pytest --dataset-loc=$VAL_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE

export RESULTS_FILE=results/test_test_data_results.txt
pytest --dataset-loc=$TEST_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE

# Test code
export RESULTS_FILE=results/test_code_results.txt
python -m pytest tests/code --trainset-loc=$TRAIN_LOC --valset-loc=$VAL_LOC --verbose --disable-warnings > $RESULTS_FILE

# Test model
export RESULTS_FILE=results/test_model_results.txt
pytest  tests/model --verbose --disable-warnings > $RESULTS_FILE
