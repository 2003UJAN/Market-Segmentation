# test_pipeline.py

import pandas as pd
from src.data_generation import create_synthetic_dataset
from src.segmentation_pipeline import build_pipeline, train_pipeline, predict_segments

def test_pipeline_build():
    model = build_pipeline()
    assert model is not None

def test_pipeline_train():
    df = create_synthetic_dataset(n=300)
    model = build_pipeline()
    model.fit(df)
    assert hasattr(model.named_steps["kmeans"], "cluster_centers_")

def test_predict_segments_output():
    df = create_synthetic_dataset(n=100)
    model = build_pipeline()
    model.fit(df)
    result = predict_segments(df, model)

    assert "cluster" in result.columns
    assert "pc1" in result.columns
    assert "pc2" in result.columns
    assert result["cluster"].nunique() > 1
