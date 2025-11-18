"""Utility functions for SPARQL Humanize.

This module provides various utility functions for model evaluation and file handling.
"""

from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import config
from .logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: Any, X_test: Any, y_test: Any
) -> Dict[str, Any]:
    """Evaluate model performance on test data.

    Args:
        model: Trained scikit-learn model.
        X_test: Test input data.
        y_test: Test labels (encoded).

    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("Evaluating model performance")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    logger.info("Test accuracy: %.4f", accuracy)
    logger.info("F1-score (weighted): %.4f", f1_weighted)
    logger.info("F1-score (macro): %.4f", f1_macro)

    return results


def get_model_path() -> str:
    """Get the full path for saving/loading the model.

    Returns:
        Path to model file.
    """
    model_dir = config.get("model.paths.model_dir", "models")
    model_file = config.get("model.paths.model_file", "sparql_classifier.pkl")
    return str(Path(model_dir) / model_file)


def ensure_model_dir_exists() -> None:
    """Ensure the model directory exists."""
    model_dir = Path(config.get("model.paths.model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Model directory: %s", model_dir)
