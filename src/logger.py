"""Logging configuration and utilities for SPARQL Humanize."""

import logging
import logging.config
from pathlib import Path
from typing import Optional

from .config import config


def setup_logging(log_level: Optional[str] = None) -> None:
    """Set up logging configuration from config.yaml.

    Args:
        log_level: Override log level. If None, uses config value.
    """
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Get logging configuration
    logging_config = config.get("logging", {})

    # Override log level if provided
    if log_level:
        if "root" in logging_config:
            logging_config["root"]["level"] = log_level.upper()
        if "loggers" in logging_config and "sparql_humanize" in logging_config["loggers"]:
            logging_config["loggers"]["sparql_humanize"]["level"] = log_level.upper()

    # Apply logging configuration
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Name of the logger (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()
