"""Data preprocessing module for SPARQL Humanize.

This module handles loading datasets and preprocessing text data for the model.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import config
from .i18n import _
from .logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        file_path: Path to the CSV file. If None, uses config default.

    Returns:
        DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    if file_path is None:
        file_path = config.get("data.dataset_path", "data/dataset.csv")

    logger.info(_("Loading data") + f": {file_path}")

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(_("File not found") + f": {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(_("Data loaded successfully") + f": {len(df)} rows")
        return df
    except Exception as e:
        logger.error(_("Error loading data") + f": {str(e)}")
        raise


def preprocess_data(
    df: pd.DataFrame,
    tokenizer: Optional[Tokenizer] = None,
    max_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
    """Preprocess text data for training or prediction.

    Args:
        df: DataFrame containing questions and query IDs.
        tokenizer: Pre-fitted tokenizer. If None, creates and fits a new one.
        max_length: Maximum sequence length. If None, uses config or calculates from data.

    Returns:
        Tuple of (X, y, tokenizer) where:
            - X: Padded sequences of tokenized text
            - y: One-hot encoded labels
            - tokenizer: Fitted tokenizer instance
    """
    logger.info(_("Preprocessing data"))

    question_column = config.get("data.question_column", "question")
    query_id_column = config.get("data.query_id_column", "query_id")

    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in dataset")

    # Initialize or use existing tokenizer
    if tokenizer is None:
        max_features = config.get("preprocessing.tokenizer.max_features")
        oov_token = config.get("preprocessing.tokenizer.oov_token", "<OOV>")

        tokenizer = Tokenizer(num_words=max_features, oov_token=oov_token)
        tokenizer.fit_on_texts(df[question_column])
        logger.debug(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(df[question_column])

    # Determine max_length
    if max_length is None:
        max_length = config.get("preprocessing.padding.max_length")
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        logger.debug(f"Using max_length: {max_length}")

    # Pad sequences
    padding_type = config.get("preprocessing.padding.padding_type", "post")
    truncating_type = config.get("preprocessing.padding.truncating_type", "post")

    X = pad_sequences(
        sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=truncating_type,
    )

    # One-hot encode labels
    if query_id_column in df.columns:
        y = pd.get_dummies(df[query_id_column]).values
        logger.debug(f"Number of classes: {y.shape[1]}")
    else:
        # For prediction without labels
        y = np.array([])

    logger.info(_("Data preprocessed successfully"))
    return X, y, tokenizer
