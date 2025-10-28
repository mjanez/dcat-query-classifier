"""Model architecture module for SPARQL Humanize.

This module defines the machine learning models for classifying questions.
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import config
from .i18n import _
from .logger import get_logger

logger = get_logger(__name__)


def build_model(model_type: str = "logistic_regression") -> Any:
    """Build a scikit-learn classifier for question classification.

    Args:
        model_type: Type of classifier to build. Options:
            - "logistic_regression": Logistic Regression (default)
            - "svm": Support Vector Machine
            - "random_forest": Random Forest
            - "naive_bayes": Multinomial Naive Bayes

    Returns:
        Configured scikit-learn classifier.
    """
    logger.info(_("Building model"))
    logger.info(f"Model type: {model_type}")

    # Get model configuration
    random_state = config.get("model.random_state", 42)

    # Build model based on type
    if model_type == "logistic_regression":
        max_iter = config.get("model.logistic_regression.max_iter", 1000)
        c_value = config.get("model.logistic_regression.C", 1.0)
        model = LogisticRegression(
            max_iter=max_iter, C=c_value, random_state=random_state
        )
    elif model_type == "svm":
        kernel = config.get("model.svm.kernel", "linear")
        c_value = config.get("model.svm.C", 1.0)
        model = SVC(kernel=kernel, C=c_value, random_state=random_state)
    elif model_type == "random_forest":
        n_estimators = config.get("model.random_forest.n_estimators", 100)
        max_depth = config.get("model.random_forest.max_depth", None)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
    elif model_type == "naive_bayes":
        alpha = config.get("model.naive_bayes.alpha", 1.0)
        model = MultinomialNB(alpha=alpha)
    else:
        logger.warning(f"Unknown model type: {model_type}, using logistic_regression")
        model = LogisticRegression(max_iter=1000, random_state=random_state)

    logger.info(_("Model built successfully"))
    logger.debug(f"Model: {model}")

    return model
