"""Utility functions for SPARQL Humanize.

This module provides various utility functions for model evaluation and file handling.
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import config
from .logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: Model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """Evaluate model performance on test data.

    Args:
        model: Trained Keras model.
        X_test: Test input data.
        y_test: Test labels (one-hot encoded).

    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("Evaluating model performance")

    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    logger.info("Test accuracy: %.4f", test_accuracy)
    logger.info("Test loss: %.4f", test_loss)
    logger.debug("Classification report:\n%s", classification_report(y_true, y_pred))

    return results


def save_tokenizer(tokenizer: Tokenizer, filepath: str = None) -> None:
    """Save tokenizer to disk.

    Args:
        tokenizer: Tokenizer instance to save.
        filepath: Path to save the tokenizer. If None, uses config default.
    """
    if filepath is None:
        model_dir = config.get("model.paths.model_dir", "models")
        tokenizer_file = config.get("model.paths.tokenizer_file", "tokenizer.pkl")
        filepath = str(Path(model_dir) / tokenizer_file)

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)

    logger.info("Tokenizer saved to: %s", filepath)


def load_tokenizer(filepath: str = None) -> Tokenizer:
    """Load tokenizer from disk.

    Args:
        filepath: Path to the tokenizer file. If None, uses config default.

    Returns:
        Loaded Tokenizer instance.

    Raises:
        FileNotFoundError: If tokenizer file doesn't exist.
    """
    if filepath is None:
        model_dir = config.get("model.paths.model_dir", "models")
        tokenizer_file = config.get("model.paths.tokenizer_file", "tokenizer.pkl")
        filepath = str(Path(model_dir) / tokenizer_file)

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)

    logger.info("Tokenizer loaded from: %s", filepath)
    return tokenizer


def get_model_path() -> str:
    """Get the full path for saving/loading the model.

    Returns:
        Path to model file.
    """
    model_dir = config.get("model.paths.model_dir", "models")
    model_file = config.get("model.paths.model_file", "sparql_classifier.h5")
    return str(Path(model_dir) / model_file)


def ensure_model_dir_exists() -> None:
    """Ensure the model directory exists."""
    model_dir = Path(config.get("model.paths.model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Model directory: %s", model_dir)
