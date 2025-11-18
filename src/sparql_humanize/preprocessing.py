"""Data preprocessing module for SPARQL Humanize.

This module handles loading datasets and preprocessing text data for the model.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, TfidfVectorizer, LabelEncoder]:
    """Preprocess text data for training or prediction.

    Args:
        df: DataFrame containing questions and query types.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, vectorizer, label_encoder) where:
            - X_train: Vectorized training features
            - X_test: Vectorized test features
            - y_train: Encoded training labels
            - y_test: Encoded test labels
            - vectorizer: Fitted TF-IDF vectorizer
            - label_encoder: Fitted label encoder
    """
    logger.info(_("Preprocessing data"))

    question_column = config.get("data.question_column", "question")
    query_type_column = config.get("data.query_type_column", "query_type")

    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in dataset")
    if query_type_column not in df.columns:
        raise ValueError(f"Column '{query_type_column}' not found in dataset")

    # Get data
    X = df[question_column]
    y = df[query_type_column]

    # Split data
    if random_state is None:
        random_state = config.get("model.random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Split data: train=%d, test=%d", len(X_train), len(X_test))

    # Vectorize text
    max_features = config.get("model.max_features", 5000)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    logger.info("Vectorized features: %d", X_train_vectorized.shape[1])

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    logger.info("Number of classes: %d", len(label_encoder.classes_))
    logger.info(_("Data preprocessed successfully"))

    return X_train_vectorized, X_test_vectorized, y_train_encoded, y_test_encoded, vectorizer, label_encoder
