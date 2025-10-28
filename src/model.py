"""Model architecture module for SPARQL Humanize.

This module defines the neural network architecture for classifying questions.
"""

from typing import List

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential

from .config import config
from .i18n import _
from .logger import get_logger

logger = get_logger(__name__)


def build_model(input_dim: int, max_length: int, output_dim: int) -> Sequential:
    """Build and compile the LSTM model for question classification.

    Args:
        input_dim: Size of the vocabulary (number of unique tokens).
        max_length: Maximum length of input sequences.
        output_dim: Number of output classes (SPARQL query types).

    Returns:
        Compiled Keras Sequential model.
    """
    logger.info(_("Building model"))

    # Get model configuration
    embedding_dim = config.get("model.architecture.embedding_dim", 50)
    lstm_units: List[int] = config.get("model.architecture.lstm_units", [64, 32])
    dropout_rate = config.get("model.architecture.dropout_rate", 0.5)
    dense_units = config.get("model.architecture.dense_units", 32)
    activation = config.get("model.architecture.activation", "relu")
    output_activation = config.get("model.architecture.output_activation", "softmax")

    # Build model
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_dim,
            input_length=max_length,
            name="embedding",
        )
    )

    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(LSTM(units, return_sequences=return_sequences, name=f"lstm_{i+1}"))

        if return_sequences:
            model.add(Dropout(dropout_rate, name=f"dropout_lstm_{i+1}"))

    # Dense layers
    model.add(Dense(dense_units, activation=activation, name="dense"))
    model.add(Dense(output_dim, activation=output_activation, name="output"))

    # Compile model
    optimizer = config.get("model.training.optimizer", "adam")
    loss = config.get("model.training.loss", "categorical_crossentropy")
    metrics = config.get("model.training.metrics", ["accuracy"])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(_("Model built successfully"))
    logger.debug(f"Model summary:\n{model.summary()}")

    return model
