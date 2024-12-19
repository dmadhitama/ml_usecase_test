import pytest
import pandas as pd
import numpy as np
from wholesale_analysis import load_and_preprocess_data, WholesaleDataset

def test_load_and_preprocess_data(tmp_path):
    # Create a sample CSV file
    sample_data = pd.DataFrame({
        'Channel': [1, 2],
        'Region': [1, 2],
        'Fresh': [100, 200],
        'Milk': [150, 250],
        'Grocery': [200, 300],
        'Frozen': [50, 100],
        'Detergents_Paper': [75, 125],
        'Delicassen': [25, 75]
    })
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    
    # Test the function
    df = load_and_preprocess_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert all(col in df.columns for col in sample_data.columns)

def test_wholesale_dataset():
    # Create sample data
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels = np.array([0, 1])
    
    # Create dataset
    dataset = WholesaleDataset(features, labels)
    
    # Test dataset properties
    assert len(dataset) == 2
    assert dataset[0][0].shape == (3,)  # Feature shape
    assert isinstance(dataset[0][1].item(), int)  # Label type