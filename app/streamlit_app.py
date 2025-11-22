# app.py
"""
All-in-one Streamlit app for Customer Segmentation:
- Generates synthetic dataset (or accepts uploaded CSV)
- Trains preprocessing + PCA + KMeans pipeline
- Saves/loads model to models/segmentation_pipeline.joblib
- Visualizes clusters (PCA scatter, counts, profiles)
- Predicts segment for uploaded or manual input rows
Compatible with scikit-learn >= 1.4 (OneHotEncoder(sparse_output=False))
"""

import os
import sys
from pathlib import Path
import io
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---------------------------
# Paths & ensure folders
# ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "synthetic_consumer_survey.csv"
MODEL_PATH = MODELS_DIR / "segmentation_pipeline.joblib"

# ---------------------------
# Feature definitions
# ---------------------------
NUMERICAL_COLS = [
    "Age", "Income", "FMCG_Spend", "Electronics_Spend", "Entertainment_Spend"
]
CATEGORICAL_COLS = ["City_Tier", "Brand_Preference", "Lifestyle"]
ALL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# ---------------------------
# Synthetic data generator
# ---------------------------
def create_synthetic_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, size=n)
    income = (rng.normal(60000, 25000, size=n)).clip(10000, 250000).astype(int)
    city_tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], size=n, p=[0.25, 0.45, 0.30])
    fmcg = (rng.normal(1500, 800, size=n)).clip(100, 10000).round(1)
    elec = (rng.normal(4000, 6000, size=n)).clip(0, 50000).round(1)
    ent = (rng.normal(800, 900, size=n)).clip(0, 15000).round(1)
    brand = rng.choice(["Brand-A", "Brand-B", "Brand-C", "Brand-D"], size=n, p=[0.35,0.25,0.25,0.15])
    lifestyle = rng.choice(["Budget", "Moderate", "Premium"], size=n, p=[0.4,0.45,0.15])

    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "City_Tier": city_tier,
        "FMCG_Spend": fmcg,
        "Electronics_Spend": elec,
        "Entertainment_Spend": ent,
        "Brand_Preference": brand,
        "Lifestyle": lifestyle
    })
    return df

# ---------------------------
# Pipeline builders & helpers
# ---------------------------
def build_pipeline(n_pca: int = 2, k: int = 4, random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="drop"
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("pca", PCA(n_components=n_pca, random_state=random_state)),
            ("kmeans", KMeans(n_clusters=k, random_state=random_state)),
        ]
    )
    return model

def save_model(pipeline: Pipeline, path: Path = MODEL_PATH):
    joblib.dump(pipeline, path)

def load_model(path: Path = MODEL_PATH) -> Optional[Pipeline]:
    if not path.exists():
        return None
    return joblib.load(path)

def train_and_save(df: pd.DataFrame, k: int = 4, n_pca: int = 2, save_path: Path = MODEL_PATH) -> Pipeline:
    model = build_pipeline(n_pca=n_pca, k=k)
    model.fit(df)
    save_model(model, save_path)
    return model

def predict_with_model(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    pro = model.named_steps["preprocess"].transform(df)
    pcs = model.named_steps["pca"].transform(pro)
    clusters = model.named_steps["kmeans"].predict(pro)
    out = df.copy().reset_index(drop=True)
    out["pc1"] = pcs[:, 0]
    out["pc2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0
    out["Cluster"] = clusters
    return out

def profile_clusters(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    agg_num = df_with_clusters.groupby("Cluster")[NUMERICAL_COLS].median()
    # for categorical: most frequent
    cat_mode = df_with_clusters.groupby("Cluster")[CATEGORICAL_COLS].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    profile = pd.concat([agg_num, cat_mode], axis=1).reset_index()
    # add counts
    profile["Count"] = df_with_clusters.groupby("Cluster").size().values
    cols = ["Cluster", "Count"] + [c for c in profile.columns if c not in ["Cluster", "Count"]]
    return profile[cols]

# ---------------------------
# Small plotting helpers
# ---------------------------
def plot_pca_scatter(df_with_clusters: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(df_with_clusters["pc1"], df_with_clusters["pc2"], c=df_with_clusters["Cluster"], cmap="tab10", alpha=0.7, s=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection (colored by cluster)")
    plt.tight_layout()
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: ; }
    .block { padding: 10px; border-radius: 12px; background: rgba(255,255,255,0.02); }
    .big-btn > button { padding: 0.6rem 1.2rem; border-radius: 10px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True
)

st.title("Customer Segmentation — Interactive App")
st.write("Generate data, train a segmentation model, visualize clusters, and predict segments. No extra files required.")

# Sidebar controls
st.sidebar.header("Workflow")
mode = st.sidebar.radio("Select", ["Overview", "Train", "Explore Clusters", "Predict"])

# Upload or use default
uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
if uploaded:
    try:
        df_uploaded = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df_uploaded = None
else:
    df_uploaded = None

# Quick parameters
st.sidebar.markdown("---")
k_val = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
pca_val = st.sidebar.slider("PCA components", 2, 5, 2)
st.sidebar.markdown("---")
if st.sidebar.button("Delete saved model"):
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        st.sidebar.success("Deleted saved model.")
    else:
        st.sidebar.info("No saved model found.")

# ---------------------------
# Overview
# ---------------------------
if mode == "Overview":
    st.header("Overview")
    st.markdown("- Use the **Train** tab to create a segmentation model from synthetic data or your uploaded CSV.")
    st.markdown("- Use **Explore Clusters** to visualize and profile segments.")
    st.markdown("- Use **Predict** to classify new customers (manual entry or uploaded CSV).")
    st.markdown("---")
    st.subheader("Sample synthetic dataset")
    if not DATA_PATH.exists():
        df_sample = create_synthetic_dataset(500)
        df_sample.to_csv(DATA_PATH, index=False)
    else:
        df_sample = pd.read_csv(DATA_PATH)
    st.dataframe(df_sample.head(10))
    st.download_button("Download sample CSV", df_sample.to_csv(index=False), file_name="synthetic_consumer_survey.csv")

# ---------------------------
# Train
# ---------------------------
elif mode == "Train":
    st.header("Train segmentation model")
    st.write("Provide data (upload) or use synthetic sample.")

    use_uploaded = st.checkbox("Use uploaded dataset" if df_uploaded is not None else "Upload a dataset to use it", value=False)
    if use_uploaded and df_uploaded is not None:
        df_train = df_uploaded.copy()
    else:
        n_rows = st.number_input("Synthetic rows", 200, 5000, 1000, step=100)
        if st.button("Generate synthetic dataset"):
            df_gen = create_synthetic_dataset(int(n_rows))
            df_gen.to_csv(DATA_PATH, index=False)
            st.success(f"Generated and saved {n_rows} synthetic rows to {DATA_PATH}")
        df_train = pd.read_csv(DATA_PATH)

    st.subheader("Preview training data")
    st.dataframe(df_train.head(8))

    if st.button("Train model now"):
        # validate columns
        missing = [c for c in ALL_COLS if c not in df_train.columns]
        if missing:
            st.error(f"Missing required columns for training: {missing}")
        else:
            with st.spinner("Training model…"):
                try:
                    model = train_and_save(df_train[ALL_COLS], k=k_val, n_pca=pca_val, save_path=MODEL_PATH)
                    st.success("Model trained and saved.")
                    # silhouette (optional)
                    # compute silhouette on transformed dataset if k>1
                    if k_val > 1:
                        pro = model.named_steps["preprocess"].transform(df_train[ALL_COLS])
                        labels = model.named_steps["kmeans"].labels_
                        try:
                            sil = silhouette_score(pro, labels)
                            st.metric("Silhouette score (preprocessed)", f"{sil:.3f}")
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Training failed: {e}")

# ---------------------------
# Explore Clusters
# ---------------------------
elif mode == "Explore Clusters":
    st.header("Explore clusters & profiles")
    model = load_model()
    if model is None:
        st.warning("No trained model found. Train a model first.")
    else:
        # choose data to visualize
        data_choice = st.radio("Data to visualize", ["Training data (saved)", "Upload CSV to visualize"])
        if data_choice == "Upload CSV to visualize":
            uploaded_viz = st.file_uploader("CSV for visualization", type=["csv"], key="viz")
            if uploaded_viz is None:
                st.info("Upload a CSV to visualize clusters.")
                st.stop()
            df_viz = pd.read_csv(uploaded_viz)
        else:
            if not DATA_PATH.exists():
                st.error("Saved training data not found. Re-train model or upload CSV.")
                st.stop()
            df_viz = pd.read_csv(DATA_PATH)

        missing = [c for c in ALL_COLS if c not in df_viz.columns]
        if missing:
            st.error(f"Missing columns in visualization data: {missing}")
            st.stop()

        result = predict_with_model(df_viz[ALL_COLS], model)
        st.subheader("Cluster counts")
        counts = result["Cluster"].value_counts().sort_index()
        st.bar_chart(counts)

        st.subheader("PCA scatter")
        fig = plot_pca_scatter(result)
        st.pyplot(fig)

        st.subheader("Cluster profiles (median numeric, mode categorical)")
        prof = profile_clusters(result)
        st.dataframe(prof)

        st.download_button("Download clustered CSV", result.to_csv(index=False), "clustered_output.csv", "text/csv")

# ---------------------------
# Predict
# ---------------------------
elif mode == "Predict":
    st.header("Predict segment for new customers")
    model = load_model()
    if model is None:
        st.warning("No trained model found. Train a model first.")
        st.stop()

    input_mode = st.radio("Input mode", ["Manual single input", "Upload CSV for batch prediction"])
    if input_mode == "Manual single input":
        st.subheader("Manual input")
        manual = {}
        for col in NUMERICAL_COLS:
            manual[col] = st.number_input(col, value=int(0))
        # categorical selects
        manual["City_Tier"] = st.selectbox("City_Tier", ["Tier-1", "Tier-2", "Tier-3"])
        manual["Brand_Preference"] = st.selectbox("Brand_Preference", ["Brand-A","Brand-B","Brand-C","Brand-D"])
        manual["Lifestyle"] = st.selectbox("Lifestyle", ["Budget","Moderate","Premium"])
        if st.button("Predict this customer"):
            newdf = pd.DataFrame([manual])
            pred = predict_with_model(newdf[ALL_COLS], model)
            st.success(f"Predicted cluster: {int(pred['Cluster'].iloc[0])}")
            st.dataframe(pred)
    else:
        up = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="pred")
        if up is not None:
            dfp = pd.read_csv(up)
            missing = [c for c in ALL_COLS if c not in dfp.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
            else:
                res = predict_with_model(dfp[ALL_COLS], model)
                st.dataframe(res.head(50))
                st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------
# End of app
# ---------------------------
st.markdown("---")
st.caption("Segmentation app — single-file edition. Modify feature lists at the top to match your schema.")
