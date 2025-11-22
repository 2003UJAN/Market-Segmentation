# app.py
"""
Single-file Streamlit app: Train + Predict Customer Segmentation
- Handles uploaded CSVs with column mapping (your column names -> pipeline names)
- Trains and persists model (models/segmentation_pipeline.joblib)
- Visualizes clusters (PCA scatter), cluster counts, cluster profiles
- Predicts segments (manual input or CSV batch)
Compatible with scikit-learn >= 1.4 (OneHotEncoder(sparse_output=False))
"""

import os
from pathlib import Path
import io
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------
# Uploaded sample path from session (image shown earlier)
# ---------------------------
# Developer note: this is the local path you uploaded in the session.
# It's included here so you can easily reference it if needed.
SAMPLE_UPLOADED_PATH = "/mnt/data/bc722270-3de3-4cfa-859d-6fbc2343ab6d.png"

# ---------------------------
# Paths: ensure directories exist
# ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "synthetic_consumer_survey.csv"
MODEL_PATH = MODELS_DIR / "segmentation_pipeline.joblib"

# ---------------------------
# Expected pipeline feature names (internal schema)
# ---------------------------
NUMERICAL_COLS = [
    "Age", "Income", "FMCG_Spend", "Electronics_Spend", "Entertainment_Spend"
]
CATEGORICAL_COLS = ["City_Tier", "Brand_Preference", "Lifestyle"]
ALL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

# ---------------------------
# Mapping from user's CSV to internal names
# (adjust here if your CSV uses slightly different names)
# ---------------------------
COLUMN_MAPPING = {
    "customer_id": "Customer_ID",
    "age": "Age",
    "income": "Income",
    "city_tier": "City_Tier",
    "spend_fmcg": "FMCG_Spend",
    "spend_electronics": "Electronics_Spend",
    "spend_entertainment": "Entertainment_Spend",
    "pref_brand": "Brand_Preference",
    "fitness": "Fitness",  # optional, not required by pipeline
    "family_oriented": "Family_Oriented",
    "tech_savvy": "Tech_Savvy",
    "eco_friendly": "Eco_Friendly"
}

# ---------------------------
# Utility functions
# ---------------------------
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to map common user column names (lowercase/underscore) to the required internal schema.
    Non-mapped columns are retained.
    """
    df = df.copy()
    lower_map = {col.lower().strip(): col for col in df.columns}
    rename_map = {}
    for user_col, target_col in COLUMN_MAPPING.items():
        # match by exact or case-insensitive
        if user_col in df.columns:
            rename_map[user_col] = target_col
        else:
            # try lowercase match
            if user_col.lower() in lower_map:
                rename_map[lower_map[user_col.lower()]] = target_col
    # apply rename
    df = df.rename(columns=rename_map)
    return df

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

def save_model(model: Pipeline, path: Path = MODEL_PATH):
    joblib.dump(model, path)

def load_model(path: Path = MODEL_PATH) -> Optional[Pipeline]:
    if not path.exists():
        return None
    return joblib.load(path)

def train_model(df: pd.DataFrame, k: int = 4, n_pca: int = 2) -> Pipeline:
    model = build_pipeline(n_pca=n_pca, k=k)
    model.fit(df[ALL_FEATURES])
    save_model(model)
    return model

def predict_df(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    df_in = df.copy().reset_index(drop=True)
    transformed = model.named_steps["preprocess"].transform(df_in[ALL_FEATURES])
    pcs = model.named_steps["pca"].transform(transformed)
    clusters = model.named_steps["kmeans"].predict(transformed)
    df_in["pc1"] = pcs[:, 0]
    df_in["pc2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0
    df_in["Cluster"] = clusters
    return df_in

def profile_clusters(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    agg_num = df_with_clusters.groupby("Cluster")[NUMERICAL_COLS].median()
    cat_mode = df_with_clusters.groupby("Cluster")[CATEGORICAL_COLS].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    profile = pd.concat([agg_num, cat_mode], axis=1).reset_index()
    profile["Count"] = df_with_clusters.groupby("Cluster").size().values
    cols = ["Cluster", "Count"] + [c for c in profile.columns if c not in ["Cluster", "Count"]]
    return profile[cols]

def plot_pca(df_with_clusters: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(df_with_clusters["pc1"], df_with_clusters["pc2"], c=df_with_clusters["Cluster"], cmap="tab10", alpha=0.7, s=30)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA projection (colored by cluster)")
    plt.tight_layout()
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Segmentation — Train & Predict", layout="wide")
st.title("Customer Segmentation — Train & Predict")

st.sidebar.header("Workflow")
mode = st.sidebar.radio("Mode", ["Overview", "Train", "Explore Clusters", "Predict"])

st.sidebar.markdown("**Upload a CSV** (optional) — it will be used for training if selected.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# quick controls
k = st.sidebar.slider("K (clusters)", 2, 8, 4)
n_pca = st.sidebar.slider("PCA components", 2, 3, 2)

if st.sidebar.button("Delete saved model"):
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        st.sidebar.success("Deleted saved model.")
    else:
        st.sidebar.info("No saved model to delete.")

# ---------------------------
# Overview
# ---------------------------
if mode == "Overview":
    st.header("Overview")
    st.write("- Train a KMeans segmentation model (PCA preprocessing).")
    st.write("- Explore clusters and predict new customers.")
    st.write(f"- Sample uploaded file path (for reference): `{SAMPLE_UPLOADED_PATH}`")
    st.markdown("---")
    if not DATA_PATH.exists():
        df_sample = create_synthetic_dataset(500)
        df_sample.to_csv(DATA_PATH, index=False)
    else:
        df_sample = pd.read_csv(DATA_PATH)
    st.subheader("Sample dataset (generated)")
    st.dataframe(df_sample.head(10))
    st.download_button("Download sample CSV", df_sample.to_csv(index=False), "synthetic_consumer_survey.csv", "text/csv")

# ---------------------------
# Train
# ---------------------------
elif mode == "Train":
    st.header("Train segmentation model")
    st.write("Option: upload your CSV (app will map columns if possible) or use synthetic data.")

    df_train = None
    use_uploaded = False

    if uploaded_file is not None:
        try:
            tmp = pd.read_csv(uploaded_file)
            st.write("Uploaded file columns:", tmp.columns.tolist())
            tmp = standardize_column_names(tmp)
            st.write("After mapping, columns:", tmp.columns.tolist())
            df_train = tmp
            use_uploaded = st.checkbox("Use uploaded CSV for training", value=True)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    if not use_uploaded:
        n = st.number_input("Rows for synthetic dataset", min_value=200, max_value=10000, value=1000, step=100)
        if st.button("Generate synthetic dataset"):
            df_gen = create_synthetic_dataset(n)
            df_gen.to_csv(DATA_PATH, index=False)
            st.success(f"Synthetic dataset with {n} rows created at {DATA_PATH}")
        df_train = pd.read_csv(DATA_PATH)

    st.subheader("Preview training data")
    st.dataframe(df_train.head(8))

    # Ensure training df has required columns: map if possible
    # If dataset uses lowercase names (user provided), remap automatically
    df_train = standardize_column_names(df_train)

    missing = [c for c in ALL_FEATURES if c not in df_train.columns]
    if missing:
        st.error(f"Missing required columns for training: {missing}")
        st.info("Expected columns (internal schema): " + ", ".join(ALL_FEATURES))
        st.stop()

    if st.button("Train model now"):
        try:
            model = train_model(df_train, k=k, n_pca=n_pca)
            st.success("Model trained and saved.")
            # silhouette score (on preprocessed features)
            try:
                pre = model.named_steps["preprocess"].transform(df_train[ALL_FEATURES])
                labels = model.named_steps["kmeans"].labels_
                sil = silhouette_score(pre, labels)
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
        st.warning("No trained model found — train first.")
        st.stop()

    # Let user choose a file to visualize or use saved training data
    choice = st.radio("Data to visualize", ("Saved training data", "Upload CSV to visualize"))
    if choice == "Upload CSV to visualize":
        up = st.file_uploader("Upload CSV for visualization", type=["csv"], key="viz")
        if up is None:
            st.info("Upload a CSV to visualize clusters.")
            st.stop()
        df_viz = pd.read_csv(up)
        df_viz = standardize_column_names(df_viz)
    else:
        if not DATA_PATH.exists():
            st.error("Saved training data not found. Re-train or upload data.")
            st.stop()
        df_viz = pd.read_csv(DATA_PATH)

    missing = [c for c in ALL_FEATURES if c not in df_viz.columns]
    if missing:
        st.error(f"Missing columns in visualization data: {missing}")
        st.stop()

    result = predict_df(df_viz[ALL_FEATURES], model)
    st.subheader("Cluster counts")
    st.bar_chart(result["Cluster"].value_counts().sort_index())

    st.subheader("PCA scatter")
    fig = plot_pca(result)
    st.pyplot(fig)

    st.subheader("Cluster profiles")
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
        st.warning("No trained model found — train first.")
        st.stop()

    input_mode = st.radio("Input mode", ("Manual single input", "Upload CSV for batch prediction"))
    if input_mode == "Manual single input":
        st.subheader("Manual input")
        manual = {}
        for col in NUMERICAL_COLS:
            manual[col] = st.number_input(col, min_value=0, value=int(0))
        # map categorical options reasonably
        manual["City_Tier"] = st.selectbox("City_Tier", ["Tier-1", "Tier-2", "Tier-3"])
        manual["Brand_Preference"] = st.selectbox("Brand_Preference", ["Brand-A", "Brand-B", "Brand-C", "Brand-D"])
        manual["Lifestyle"] = st.selectbox("Lifestyle", ["Budget", "Moderate", "Premium"])
        if st.button("Predict this customer"):
            newdf = pd.DataFrame([manual])
            try:
                pred = predict_df(newdf[ALL_FEATURES], model)
                st.success(f"Predicted cluster: {int(pred['Cluster'].iloc[0])}")
                st.dataframe(pred)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        up2 = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="pred")
        if up2 is not None:
            dfp = pd.read_csv(up2)
            dfp = standardize_column_names(dfp)
            missing = [c for c in ALL_FEATURES if c not in dfp.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
            else:
                try:
                    res = predict_df(dfp[ALL_FEATURES], model)
                    st.dataframe(res.head(50))
                    st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Single-file segmentation app. Adjust COLUMN_MAPPING at top if your CSV uses different column names.")
