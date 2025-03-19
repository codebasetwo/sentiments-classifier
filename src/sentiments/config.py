import logging
import os
import sys
from pathlib import Path

import mlflow
from hyperopt import hp

CLASS_TO_INDEX = {"positive": 2, "neutral": 1, "negative": 0}

# Root Directory
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# Data Directories
METADATA_PATH = Path(ROOT_DIR, "data/metadata")
METADATA_PATH.mkdir(parents=True, exist_ok=True)

# Data Directories
DATASET_PATH = Path(ROOT_DIR, "data/datasets")
DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Config Mlflow
MLFLOW_DIR = Path(
    f"/tmp/{os.environ.get('GITHUB_USERNAME', 'codebasetwo')}/mlflow"
).absolute()

Path(MLFLOW_DIR).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file:///" + str(MLFLOW_DIR)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# logs
LOGS_DIR = Path("/tmp/logs/").absolute()
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# Hyperparameter search space
SPACE = {
    "learning_rate": hp.choice("learning_rate", [0.001, 0.0001, 0.1]),
    "num_epochs": hp.choice("num_epochs", [3, 5, 7, 10]),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
    "vocab_size": hp.choice("vocab_size", [5000, 1000, 3000, 2000]),
}

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
