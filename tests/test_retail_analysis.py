import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from retail_analysis import load_and_preprocess_data, calculate_rfm_metrics

def test_load_and_preprocess_data(tmp_path):
    # Create a sample Excel file
    sample_data = pd.DataFrame({
        'InvoiceNo': ['A123', 'B456'],
        'CustomerID': [1, 2],
        'InvoiceDate': [datetime.now(), datetime.now()],
        'Quantity': [2, 3],
        'UnitPrice': [10.0, 15.0]
    })
    file_path = tmp_path / "test_data.xlsx"
    sample_data.to_excel(file_path, index=False)
    
    # Test the function
    df = load_and_preprocess_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'TotalAmount' in df.columns
    assert all(df['TotalAmount'] == df['Quantity'] * df['UnitPrice'])

def test_calculate_rfm_metrics():
    # Create sample data
    sample_data = pd.DataFrame({
        'CustomerID': [1, 1, 2],
        'InvoiceDate': [
            datetime(2023, 1, 1),
            datetime(2023, 1, 15),
            datetime(2023, 2, 1)
        ],
        'InvoiceNo': ['A1', 'A2', 'B1'],
        'TotalAmount': [100.0, 150.0, 200.0]
    })
    
    # Test the function
    rfm = calculate_rfm_metrics(sample_data)
    assert isinstance(rfm, pd.DataFrame)
    assert len(rfm) == 2  # Two unique customers
    assert all(col in rfm.columns for col in ['Recency', 'Frequency', 'Monetary'])