# utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def check_file(path):
    """Check if file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return True


def load_csv(path):
    """Load CSV safely with validation."""
    check_file(path)
    return pd.read_csv(path)


def plot_pca_scatter(df):
    """Plot PCA scatter (pc1 vs pc2) colored by cluster."""
    if "pc1" not in df.columns or "pc2" not in df.columns:
        raise ValueError("PCA columns missing. Run predict_segments() first.")

    plt.figure(figsize=(6, 4))
    plt.scatter(df["pc1"], df["pc2"], c=df["cluster"], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Cluster Scatter")
    plt.grid(True)
    plt.tight_layout()
    return plt
