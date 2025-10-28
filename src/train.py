"""Training script for SPARQL Humanize model.

This script handles the complete training pipeline including data loading,
preprocessing, model building, training, and saving.
"""

import sys
from typing import Optional

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .config import config
from .i18n import _
from .logger import get_logger
from .model import build_model
from .preprocessing import load_data, preprocess_data
from .utils import ensure_model_dir_exists, get_model_path, save_tokenizer

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
        X, y, tokenizer = preprocess_data(df)
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

        # Save tokenizer
        save_tokenizer(tokenizer)

        # Build model
        input_dim = len(tokenizer.word_index) + 1
        max_length = X.shape[1]
        output_dim = y.shape[1]

        model = build_model(input_dim=input_dim, max_length=max_length, output_dim=output_dim)

        # Training configuration
        epochs = config.get("model.training.epochs", 10)
        batch_size = config.get("model.training.batch_size", 2)
        validation_split = config.get("model.training.validation_split", 0.2)

        # Callbacks
        callbacks = []

        # Early stopping
        early_stopping_config = config.get("model.training.early_stopping", {})
        if early_stopping_config:
            early_stopping = EarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_loss"),
                patience=early_stopping_config.get("patience", 3),
                restore_best_weights=early_stopping_config.get(
                    "restore_best_weights", True
                ),
                verbose=1,
            )
            callbacks.append(early_stopping)
            logger.info("Early stopping enabled")

        # Model checkpoint
        if model_output_path is None:
            model_output_path = get_model_path()

        checkpoint = ModelCheckpoint(
            filepath=model_output_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint)

        # Train model
        logger.info(_("Training model"))
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

        history = model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Save final model
        model.save(model_output_path)
        logger.info(_("Model saved") + f": {model_output_path}")

        # Log training results
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")

        logger.info(_("Model trained successfully"))

    except Exception as e:
        logger.error(_("Error training model") + f": {str(e)}", exc_info=True)
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
