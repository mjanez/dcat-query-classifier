"""Training script for SPARQL Humanize model.

This script handles the complete training pipeline including data loading,
preprocessing, model building, training, and saving.
"""

import sys
from typing import Optional, Tuple, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .config import config
from .i18n import _
from .logger import get_logger
from .model import build_model
from .preprocessing import load_data, preprocess_data
from .utils import ensure_model_dir_exists, get_model_path

logger = get_logger(__name__)


def train_model(
    dataset_path: Optional[str] = None,
    model_output_path: Optional[str] = None,
) -> None:
    """Train the SPARQL question classifier model.

    Args:
        dataset_path: Path to the training dataset CSV file.
        model_output_path: Path to save the trained model.
    """
    try:
        logger.info(_("Starting training process"))

        # Ensure model directory exists
        ensure_model_dir_exists()

        # Load data
        df = load_data(dataset_path)

        # Preprocess data
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(df)
        logger.info("Training data shape: X_train=%s, y_train=%s", X_train.shape, y_train.shape)
        logger.info("Test data shape: X_test=%s, y_test=%s", X_test.shape, y_test.shape)

        # Get model type from config
        model_type = config.get("model.type", "logistic_regression")
        
        # Build model
        model = build_model(model_type=model_type)

        # Train model
        logger.info(_("Training model"))
        logger.info("Model type: %s", model_type)

        model.fit(X_train, y_train)

        # Evaluate on test set
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info("Training accuracy: %.4f", train_score)
        logger.info("Test accuracy: %.4f", test_score)

        # Save model, vectorizer, and label encoder
        if model_output_path is None:
            model_output_path = get_model_path()

        import joblib
        from pathlib import Path
        
        model_dir = Path(model_output_path).parent
        model_name = Path(model_output_path).stem
        
        # Save all components
        joblib.dump(model, model_output_path)
        joblib.dump(vectorizer, model_dir / f"{model_name}_vectorizer.pkl")
        joblib.dump(label_encoder, model_dir / f"{model_name}_label_encoder.pkl")
        
        logger.info("Model saved: %s", model_output_path)
        logger.info("Vectorizer saved: %s", model_dir / f"{model_name}_vectorizer.pkl")
        logger.info("Label encoder saved: %s", model_dir / f"{model_name}_label_encoder.pkl")

        logger.info(_("Model trained successfully"))

    except Exception as e:
        logger.error("Error training model: %s", str(e), exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for training script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SPARQL Humanize question classifier"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to training dataset CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging level
    from .logger import setup_logging

    setup_logging(args.log_level)

    # Train model
    train_model(dataset_path=args.dataset, model_output_path=args.output)


if __name__ == "__main__":
    main()
