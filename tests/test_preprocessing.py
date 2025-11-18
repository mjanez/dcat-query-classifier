"""Test the preprocessing module."""

import pandas as pd
from sparql_humanize.preprocessing import load_data, preprocess_data

def test_load_data_csv(tmp_path):
    """Test loading data from CSV file."""
    # Create test CSV file
    test_data = pd.DataFrame({
        'question': ['What is this?', 'How to do that?'],
        'query_type': ['type1', 'type2']
    })
    csv_file = tmp_path / "test.csv"
    test_data.to_csv(csv_file, index=False)
    
    # Test loading
    result = load_data(str(csv_file))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'question' in result.columns
    assert 'query_type' in result.columns

def test_preprocess_data(sample_data):
    """Test data preprocessing."""
    df = pd.DataFrame(sample_data, columns=['question', 'query_type'])
    
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.5)
    
    assert len(X_train) == 2
    assert len(X_test) == 2
    assert len(y_train) == 2
    assert len(y_test) == 2