"""Configuration module for SPARQL Humanize.

This module handles loading and accessing configuration from config.yaml.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration class for accessing application settings."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        """Singleton pattern to ensure only one config instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if not self._config:  # Only load if not already loaded
            if config_path is None:
                config_path = self._get_default_config_path()
            self._load_config(config_path)

    @staticmethod
    def _get_default_config_path() -> str:
        """Get the default configuration file path.

        Returns:
            Path to config.yaml in the project root.
        """
        # Go up from src/sparql_humanize/ to project root
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config.yaml")

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Override with environment variables if present
        self._override_from_env()

    def _override_from_env(self) -> None:
        """Override configuration values from environment variables."""
        # Example: SPARQL_HUMANIZE_MODEL_EPOCHS overrides model.training.epochs
        env_prefix = "SPARQL_HUMANIZE_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().split("_")
                self._set_nested_value(config_key, value)

    def _set_nested_value(self, keys: list, value: str) -> None:
        """Set a nested configuration value.

        Args:
            keys: List of keys representing the path to the value.
            value: The value to set.
        """
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = self._convert_type(value)

    @staticmethod
    def _convert_type(value: str) -> Any:
        """Convert string value to appropriate type.

        Args:
            value: String value to convert.

        Returns:
            Converted value (int, float, bool, or str).
        """
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key in dot notation (e.g., 'model.training.epochs').
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Complete configuration dictionary.
        """
        return self._config.copy()

    def reload(self, config_path: Optional[str] = None) -> None:
        """Reload configuration from file.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self._config = {}
        if config_path is None:
            config_path = self._get_default_config_path()
        self._load_config(config_path)


# Global config instance
config = Config()
