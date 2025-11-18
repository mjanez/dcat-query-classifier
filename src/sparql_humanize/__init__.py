"""SPARQL Humanize - Natural Language to SPARQL Query Classifier.

This package provides tools for classifying natural language questions
into predefined SPARQL query types for DCAT datasets.
"""

__version__ = "0.1.0"
__author__ = "mjanez"
__email__ = "mjanez@example.com"

from .config import config
from .i18n import get_i18n
from .logger import get_logger, setup_logging
from .model import build_model
from .predict import predict
from .preprocessing import load_data, preprocess_data
from .train import train_model
from .utils import evaluate_model

__all__ = [
    "config",
    "get_i18n",
    "get_logger",
    "setup_logging",
    "build_model",
    "predict",
    "load_data",
    "preprocess_data",
    "train_model",
    "evaluate_model",
]
