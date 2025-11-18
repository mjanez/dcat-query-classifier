"""Command-line interface for SPARQL Humanize."""

import sys

from .train import main as train_main
from .predict import main as predict_main


def train():
    """CLI entry point for training."""
    # Remove 'train' from sys.argv before calling train_main
    sys.argv.pop(1)
    train_main()


def predict():
    """CLI entry point for prediction."""
    # Remove 'predict' from sys.argv before calling predict_main
    sys.argv.pop(1)
    predict_main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict()
    else:
        print("Usage: sparql-humanize [train|predict] [options]")
        sys.exit(1)
