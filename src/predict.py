"""Prediction script for SPARQL Humanize model.

This script handles loading the trained model and making predictions on new questions.
"""

import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .config import config
from .i18n import _
from .logger import get_logger
from .preprocessing import preprocess_data
from .utils import get_model_path, load_tokenizer

logger = get_logger(__name__)


def predict(
    question: str,
    model_path: Optional[str] = None,
    return_probabilities: bool = False,
) -> Dict[str, any]:
    """Predict SPARQL query ID for a natural language question.

    Args:
        question: Natural language question to classify.
        model_path: Path to the trained model. If None, uses config default.
        return_probabilities: Whether to return probability scores for all classes.

    Returns:
        Dictionary containing:
            - predicted_query_id: Predicted query class ID
            - confidence: Confidence score for the prediction
            - probabilities: (optional) All class probabilities
    """
    try:
        logger.info(_("Making prediction"))
        logger.debug(f"Question: {question}")

        # Load model
        if model_path is None:
            model_path = get_model_path()

        logger.info(_("Loading model") + f": {model_path}")
        model = load_model(model_path)
        logger.info(_("Model loaded successfully"))

        # Load tokenizer
        tokenizer = load_tokenizer()

        # Preprocess question
        logger.info(_("Processing question"))
        df = pd.DataFrame({config.get("data.question_column", "question"): [question]})
        X, _, _ = preprocess_data(df, tokenizer=tokenizer)

        # Make prediction
        predictions = model.predict(X, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        result = {
            "predicted_query_id": predicted_class,
            "confidence": confidence,
        }

        if return_probabilities:
            result["probabilities"] = predictions[0].tolist()

        logger.info(_("Prediction completed"))
        logger.info(
            f"{_('Predicted SPARQL query ID')}: {predicted_class} "
            f"({_('Confidence score')}: {confidence:.4f})"
        )

        return result

    except Exception as e:
        logger.error(_("Error making prediction") + f": {str(e)}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for prediction script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict SPARQL query for natural language question"
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to classify",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Show probabilities for all classes",
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

    # Make prediction
    result = predict(
        question=args.question,
        model_path=args.model,
        return_probabilities=args.probabilities,
    )

    # Print results
    print(f"\n{_('Predicted SPARQL query ID')}: {result['predicted_query_id']}")
    print(f"{_('Confidence score')}: {result['confidence']:.4f}")

    if args.probabilities and "probabilities" in result:
        print("\nClass probabilities:")
        for i, prob in enumerate(result["probabilities"]):
            print(f"  Class {i}: {prob:.4f}")


if __name__ == "__main__":
    main()
