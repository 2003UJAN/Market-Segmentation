# test_data_generation.py

import pandas as pd
from src.data_generation import create_synthetic_dataset

def test_dataset_shape():
    df = create_synthetic_dataset(n=200)
    assert df.shape[0] == 200
    assert df.shape[1] >= 10     # at least 10 columns

def test_column_presence():
    df = create_synthetic_dataset(n=50)
    required_cols = [
        "customer_id", "age", "income", "city_tier",
        "spend_fmcg", "spend_electronics", "spend_entertainment",
        "pref_brand", "fitness", "family_oriented",
        "tech_savvy", "eco_friendly"
    ]
    for col in required_cols:
        assert col in df.columns

def test_value_ranges():
    df = create_synthetic_dataset(n=100)
    assert df["age"].min() >= 18
    assert df["income"].min() >= 15000
    assert df["spend_fmcg"].max() <= 100
