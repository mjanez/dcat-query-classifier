"""Prediction script for SPARQL Humanize model.

This script handles loading the trained model and making predictions on new questions.
"""

import sys
from typing import Dict, Optional, Any
import joblib
from pathlib import Path

from .i18n import _
from .logger import get_logger
from .utils import get_model_path
from .sparql_generator import SPARQLGenerator

logger = get_logger(__name__)


def predict(
    question: str,
    model_path: Optional[str] = None,
    return_probabilities: bool = False,
    generate_sparql: bool = True,
) -> Dict[str, Any]:
    """Predict SPARQL query type for a natural language question.

    Args:
        question: Natural language question to classify.
        model_path: Path to the trained model. If None, uses config default.
        return_probabilities: Whether to return probability scores for all classes.
        generate_sparql: Whether to generate the complete SPARQL query.

    Returns:
        Dictionary containing:
            - predicted_query_type: Predicted query type
            - confidence: Confidence score for the prediction
            - probabilities: (optional) All class probabilities
            - sparql: (optional) Generated SPARQL query
            - description: (optional) Query description
            - params: (optional) Extracted parameters
    """
    try:
        logger.info(_("Making prediction"))
        logger.debug("Question: %s", question)

        # Load model components
        if model_path is None:
            model_path = get_model_path()

        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem

        logger.info("Loading model: %s", model_path)
        model = joblib.load(model_path)
        vectorizer = joblib.load(model_dir / f"{model_name}_vectorizer.pkl")
        label_encoder = joblib.load(model_dir / f"{model_name}_label_encoder.pkl")
        logger.info(_("Model loaded successfully"))

        # Vectorize question
        logger.info(_("Processing question"))
        X = vectorizer.transform([question])

        # Make prediction
        predicted_class_encoded = model.predict(X)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]
        
        # Get confidence (probability)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = float(max(probabilities))
        else:
            # For models without probability estimates
            confidence = 1.0
            probabilities = None

        result = {
            "predicted_query_type": predicted_class,
            "confidence": confidence,
        }

        if return_probabilities and probabilities is not None:
            # Map probabilities to class names
            class_probs = {
                label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            result["probabilities"] = class_probs

        logger.info(_("Prediction completed"))
        logger.info(
            "Predicted query type: %s (Confidence: %.4f)",
            predicted_class,
            confidence
        )

        # Generate SPARQL query if requested
        if generate_sparql:
            logger.info("Generating SPARQL query")
            generator = SPARQLGenerator()
            sparql_result = generator.generate(predicted_class, question)
            result.update(sparql_result)

        return result

    except Exception as e:
        logger.error("Error making prediction: %s", str(e), exc_info=True)
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
        "--no-sparql",
        action="store_true",
        help="Do not generate SPARQL query (only predict type)",
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
        generate_sparql=not args.no_sparql,
    )

    # Print results
    print(f"\n{_('Predicted SPARQL query type')}: {result['predicted_query_type']}")
    print(f"{_('Confidence score')}: {result['confidence']:.4f}")

    if "description" in result:
        print(f"{_('Description')}: {result['description']}")

    if "sparql" in result:
        print(f"\n{_('Generated SPARQL Query')}:")
        print("=" * 80)
        print(result['sparql'])
        print("=" * 80)

    if "params" in result and result["params"]:
        print(f"\n{_('Extracted Parameters')}:")
        for key, value in result["params"].items():
            print(f"  {key}: {value}")

    if args.probabilities and "probabilities" in result:
        print("\nClass probabilities:")
        for class_name, prob in result["probabilities"].items():
            print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()
