# SPARQL Humanize

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package that uses machine learning to classify natural language questions into predefined SPARQL query types for DCAT datasets. This tool helps bridge the gap between human language and structured SPARQL queries by automatically determining the intent behind natural language questions.

## Features

- **Machine Learning Classification**: Uses scikit-learn to classify natural language questions
- **SPARQL Query Type Recognition**: Identifies common DCAT dataset query patterns
- **Internationalization**: Support for multiple languages (English, Spanish)
- **Configurable**: YAML-based configuration for easy customization  
- **Logging**: Comprehensive logging with configurable levels
- **CLI Interface**: Command-line tools for training and prediction
- **Data Processing**: Built-in data preprocessing and evaluation utilities
- **Modern Python**: Type hints, proper packaging with PDM

## Quick Start

### Installation with PDM
```bash
# Clone the repository
git clone https://github.com/mjanez/sparql-humanize.git
cd sparql-humanize

# Install PDM if you haven't already
pip install pdm

# Install dependencies
pdm install

# Run commands with PDM (no need to activate virtual environment)
pdm run python -m sparql_humanize.cli --help

# Or get the virtual environment path and activate it manually
pdm info --packages  # Shows the venv path
# Then activate using: source .venv/bin/activate (Linux/Mac) or .venv\Scripts\activate (Windows)
```

### Basic Usage

#### Command Line Interface

```bash
# Using PDM
pdm run python -m sparql_humanize.cli train --dataset data/dataset.csv

# Or if you have activated the virtual environment
python -m sparql_humanize.cli train --dataset data/dataset.csv

# Make predictions
pdm run python -m sparql_humanize.cli predict "What datasets are available?"
```

#### Python API

```python
from sparql_humanize import predict, train_model
from sparql_humanize.preprocessing import load_data, preprocess_data

# Load and preprocess training data
data = load_data("data/dataset.csv")
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train a model
model, vectorizer, label_encoder = train_model(X_train, y_train)

# Make predictions
result = predict("What datasets are available?", model, vectorizer, label_encoder)
print(f"Predicted query type: {result}")
```

## Supported Query Types

The system currently recognizes these DCAT dataset query patterns:

- **`list_datasets`**: Questions asking for available datasets
- **`get_distribution`**: Questions about data distributions and formats
- **`search_datasets`**: Questions searching for specific topics or keywords
- **`get_metadata`**: Questions about dataset metadata and descriptions
- **`filter_datasets`**: Questions with filtering criteria
- **`get_statistics`**: Questions about dataset statistics and metrics

## Configuration

The project uses YAML configuration files. Copy and modify `config.yaml.example` to `config.yaml`:

```yaml
model:
  type: "text_classification"
  max_features: 5000
  random_state: 42

data:
  train_size: 0.8
  validation_size: 0.2
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
i18n:
  default_language: "en"
  supported_languages: ["en", "es"]
```

## Internationalization

The package supports multiple languages:

```python
# English (default)
python -m sparql_humanize.cli predict "What datasets are available?"

# Spanish  
python -m sparql_humanize.cli predict "¿Qué conjuntos de datos están disponibles?"
```

Add new languages by creating translation files in `locales/{lang}/LC_MESSAGES/`.

## Testing

Run the test suite:

```bash
# Using PDM
pdm run pytest

# Using pip
pip install pytest
pytest
```

Run with coverage:

```bash
pdm run pytest --cov=src/sparql_humanize --cov-report=html
```

## Model Performance

The model performance depends on your training data quality. Use the evaluation utilities:

```python
from sparql_humanize.utils import evaluate_model

# Evaluate model performance
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-score: {metrics['f1_weighted']:.4f}")
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/mjanez/sparql-humanize.git
cd sparql-humanize

# Install development dependencies
pdm install --dev

# Install pre-commit hooks
pdm run pre-commit install

# Run tests
pdm run pytest

# Run linting
pdm run ruff check src/
pdm run black src/ tests/

# Run type checking
pdm run mypy src/
```

### Working with PDM

PDM doesn't have a `shell` command. Here are the correct ways to work with PDM:

```bash
# Option 1: Use 'pdm run' prefix (recommended)
pdm run python -m sparql_humanize.cli --help
pdm run pytest
pdm run black src/

# Option 2: Manually activate the virtual environment
# First, find where PDM created the venv
pdm info --packages

# Then activate it (the path will be shown in the pdm info output)
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Now you can run commands directly
python -m sparql_humanize.cli --help
pytest

# Option 3: Use PDM scripts (defined in pyproject.toml)
pdm run test          # Runs pytest
pdm run lint          # Runs ruff
pdm run format        # Runs black
```

### Project Structure

- `src/sparql_humanize/`: Main package code
- `tests/`: Test suite
- `data/`: Training data and datasets  
- `locales/`: Translation files
- `docs/`: Documentation (if applicable)
- `config.yaml.example`: Main configuration file

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based in [miroslavliska/DcatQueryClassifier](https://github.com/miroslavliska/DcatQueryClassifier) to enhance SPARQL query classification.
- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Uses [PDM](https://pdm.fming.dev/) for modern Python package management
- Internationalization powered by [Babel](https://babel.pocoo.org/)

---

Built with ❤️ for the open data community

