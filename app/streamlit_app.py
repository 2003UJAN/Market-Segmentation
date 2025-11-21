# streamlit_app.py

import sys
import os

# Add project ROOT directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib
import os

from src.data_generation import create_synthetic_dataset
from src.segmentation_pipeline import (
    train_pipeline,
    load_pipeline,
    predict_segments,
    profile_clusters,
)
from src.utils import plot_pca_scatter

DATA_PATH = "data/synthetic_consumer_survey.csv"
MODEL_PATH = "models/segmentation_pipeline.joblib"

st.set_page_config(
    page_title="Market Segmentation Dashboard",
    layout="wide",
)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

action = st.sidebar.radio(
    "Select Action",
    ["View Dataset", "Train Model", "Cluster Visualization", "Predict Segment"],
)

st.sidebar.markdown("---")
st.sidebar.info("Hansa Research – Consumer Insights & Segmentation Demo")


# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        df = create_synthetic_dataset(2000)
        df.to_csv(DATA_PATH, index=False)
        return df


df = load_dataset()


# -------------------------------------------------------------------
# Load Model
# -------------------------------------------------------------------
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_pipeline(MODEL_PATH)
    else:
        st.warning("Model not found. Please train model first.")
        return None


# -------------------------------------------------------------------
# PAGE 1: View Dataset
# -------------------------------------------------------------------
if action == "View Dataset":
    st.title("📊 Synthetic Consumer Survey Dataset")

    st.write("This dataset is auto-generated for segmentation analysis.")
    st.dataframe(df.head(20))

    st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    st.download_button(
        "📥 Download Dataset",
        data=df.to_csv(index=False),
        file_name="synthetic_consumer_survey.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------------
# PAGE 2: Train Model
# -------------------------------------------------------------------
elif action == "Train Model":
    st.title("🤖 Train Segmentation Model")

    k_val = st.slider("Number of Segments (K-Means)", 3, 7, 4)
    pca_val = st.slider("PCA Components", 2, 3, 2)

    if st.button("🚀 Train Model"):
        model = train_pipeline(df, k=k_val, n_pca=pca_val, save_path=MODEL_PATH)
        st.success("Model trained and saved successfully!")
        st.code(f"Saved at: {MODEL_PATH}")

    st.info("Training uses PCA + KMeans + preprocessing pipeline.")


# -------------------------------------------------------------------
# PAGE 3: Cluster Visualization
# -------------------------------------------------------------------
elif action == "Cluster Visualization":
    st.title("🎯 Cluster Visualization & Profiles")

    model = get_model()
    if model is None:
        st.stop()

    st.subheader("📌 PCA Scatter Plot (PC1 vs PC2)")
    result = predict_segments(df, model)

    plt = plot_pca_scatter(result)
    st.pyplot(plt)

    st.subheader("📊 Cluster Profiles")
    profile = profile_clusters(df, model)
    st.dataframe(profile)


# -------------------------------------------------------------------
# PAGE 4: Predict Segment for New Data
# -------------------------------------------------------------------
elif action == "Predict Segment":
    st.title("🔮 Predict Segment for New Consumers")

    model = get_model()
    if model is None:
        st.stop()

    uploaded = st.file_uploader("Upload CSV (same columns as training data)", type="csv")

    if uploaded:
        new_df = pd.read_csv(uploaded)
        pred = predict_segments(new_df, model)

        st.success("Prediction Complete!")
        st.dataframe(pred[["pc1", "pc2", "cluster"] + list(new_df.columns)])

        st.download_button(
            "📥 Download Predictions",
            data=pred.to_csv(index=False),
            file_name="segment_predictions.csv",
            mime="text/csv"
        )
