import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import plotly.express as px
import os

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Engine",
    layout="wide",
    page_icon="📊"
)

MODEL_PATH = "models/segmentation_pipeline.joblib"

# ------------------------------------------------------
# COLUMN MAPPING (Your Dataset → Internal Schema)
# ------------------------------------------------------
COLUMN_MAPPING = {
    "customer_id": "Customer_ID",
    "age": "Age",
    "income": "Income",
    "city_tier": "City_Tier",
    "spend_fmcg": "FMCG_Spend",
    "spend_electronics": "Electronics_Spend",
    "spend_entertainment": "Entertainment_Spend",
    "pref_brand": "Brand_Preference",
    "fitness": "Fitness",
    "family_oriented": "Family_Oriented",
    "tech_savvy": "Tech_Savvy",
    "eco_friendly": "Eco_Friendly"
}

# -------- REQUIRED COLUMNS (Lifestyle REMOVED) ----------
REQUIRED_FEATURES = [
    "Age", "Income", "FMCG_Spend", "Electronics_Spend",
    "Entertainment_Spend", "City_Tier", "Brand_Preference"
]

# ------------------------------------------------------
# UTIL FUNCTIONS
# ------------------------------------------------------
def standardize_column_names(df):
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = df.rename(columns=COLUMN_MAPPING)
    return df


def validate_columns(df):
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing required columns for training: {missing}")
        return False
    return True


# ------------------------------------------------------
# PIPELINE (NO LIFESTYLE)
# ------------------------------------------------------
def build_pipeline(k=4, n_pca=2):

    numeric = ["Age", "Income", "FMCG_Spend", "Electronics_Spend", "Entertainment_Spend"]
    categorical = ["City_Tier", "Brand_Preference"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        ]
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("pca", PCA(n_components=n_pca)),
        ("kmeans", KMeans(n_clusters=k, random_state=42))
    ])

    return pipeline


# ------------------------------------------------------
# TRAIN & SAVE / LOAD MODEL
# ------------------------------------------------------
def train_model(df, k, n_pca):
    model = build_pipeline(k=k, n_pca=n_pca)
    model.fit(df[REQUIRED_FEATURES])
    return model

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    st.success("Model saved successfully.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("No trained model found. Train the model first.")
        return None
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------
st.title("📊 Customer Market Segmentation Engine (No Lifestyle)")

tab1, tab2, tab3 = st.tabs(["Train Model", "Cluster Visualization", "Predict Segment"])


# ------------------------------------------------------
# TAB 1 — TRAINING
# ------------------------------------------------------
with tab1:
    st.header("Train Segmentation Model")

    uploaded = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df = standardize_column_names(df)

        if validate_columns(df):
            st.success("Dataset validated successfully.")

            k = st.slider("Number of Clusters", 2, 10, 4)
            n_pca = st.slider("PCA Components", 2, 5, 2)

            if st.button("Train Model"):
                model = train_model(df, k, n_pca)
                save_model(model)
                st.success("Training complete.")


# ------------------------------------------------------
# TAB 2 — VISUALIZATION
# ------------------------------------------------------
with tab2:
    st.header("Cluster Visualization")

    model = load_model()
    if model:
        uploaded2 = st.file_uploader("Upload dataset used for training (CSV)", type=["csv"])

        if uploaded2:
            df = pd.read_csv(uploaded2)
            df = standardize_column_names(df)

            if validate_columns(df):

                transformed = model.named_steps["prep"].transform(df[REQUIRED_FEATURES])
                pca_output = model.named_steps["pca"].transform(transformed)
                clusters = model.named_steps["kmeans"].predict(pca_output)

                df_plot = pd.DataFrame({
                    "PC1": pca_output[:, 0],
                    "PC2": pca_output[:, 1],
                    "Cluster": clusters
                })

                fig = px.scatter(
                    df_plot, x="PC1", y="PC2", color="Cluster",
                    title="Cluster Visualization",
                    width=900, height=500
                )
                st.plotly_chart(fig)

                st.subheader("Cluster Summary")
                st.dataframe(
                    df.assign(Cluster=clusters).groupby("Cluster").mean()
                )


# ------------------------------------------------------
# TAB 3 — PREDICTION
# ------------------------------------------------------
with tab3:
    st.header("Predict Segment for New Customers")

    model = load_model()

    if model:
        uploaded3 = st.file_uploader("Upload new customer data (CSV)", type=["csv"])

        if uploaded3:
            df_new = pd.read_csv(uploaded3)
            df_new = standardize_column_names(df_new)

            if validate_columns(df_new):
                transformed_new = model.named_steps["prep"].transform(df_new[REQUIRED_FEATURES])
                preds = model.named_steps["kmeans"].predict(
                    model.named_steps["pca"].transform(transformed_new)
                )

                df_out = df_new.copy()
                df_out["Predicted_Segment"] = preds

                st.success("Prediction complete.")
                st.dataframe(df_out)

                csv = df_out.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv")
