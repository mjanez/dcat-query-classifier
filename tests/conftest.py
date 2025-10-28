"""Test configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
model:
  type: "text_classification"
  max_features: 5000
  random_state: 42

data:
  train_size: 0.8
  validation_size: 0.2

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""")
        yield f.name
    Path(f.name).unlink()

@pytest.fixture
def sample_data():
    """Sample training data for testing."""
    return [
        ("What datasets are available?", "list_datasets"),
        ("Show me the distribution data", "get_distribution"),
        ("Find datasets about health", "search_datasets"),
        ("Get dataset metadata", "get_metadata"),
    ]