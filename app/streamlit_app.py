# app.py
"""
Final single-file Streamlit app (Option B: Lifestyle Score)
- Robust column mapping based on uploaded CSV header variants
- Lifestyle_Score = fitness + family_oriented + tech_savvy + eco_friendly
- Pipeline: StandardScaler + OneHotEncoder + PCA + KMeans
- Save/load model to models/segmentation_pipeline.joblib
- Matplotlib-based PCA scatter (no Plotly)
"""

import os
from pathlib import Path
import sys

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

# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "segmentation_pipeline.joblib"

# -----------------------
# App config
# -----------------------
st.set_page_config(title="Customer Segmentation (Lifestyle Score)", layout="wide")
st.title("Customer Segmentation — Train & Predict")

st.markdown("**Reference uploaded file (screenshot):** `/mnt/data/33bd30a9-1d12-4a78-b5fd-aafd6625f06c.png`")

# -----------------------
# Utility: smart column matching
# -----------------------
def smart_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common user column names to internal canonical names.
    Uses case-insensitive matching and substring heuristics.
    """
    col_map = {}
    cols = list(df.columns)
    lowered = {c.lower().strip(): c for c in cols}

    def find(key_subs):
        # key_subs: list of substrings, return actual column name if any contains them
        for k in key_subs:
            for low, orig in lowered.items():
                if k in low:
                    return orig
        return None

    # mapping targets
    mapping_targets = {
        "Customer_ID": ["customer_id", "customer", "cust_id", "id"],
        "Age": ["age", "years"],
        "Income": ["income", "salary", "earn"],
        "City_Tier": ["city_tier", "city tier", "citytier", "city"],
        "FMCG_Spend": ["fmcg", "fmcg_spend", "spend_fmcg", "fmcg spend", "fmcg_sp"],
        "Electronics_Spend": ["electronics", "spend_electronics", "electronic", "elec_spend"],
        "Entertainment_Spend": ["entertain", "spend_entertainment", "entertainment_spend", "ent_spend"],
        "Brand_Preference": ["brand", "pref_brand", "preference", "brand_pref", "brand_pre"],
        "fitness": ["fitness", "fit"],
        "family_oriented": ["family", "family_oriented", "familyoriented"],
        "tech_savvy": ["tech_savvy", "tech", "techsavvy"],
        "eco_friendly": ["eco_friendly", "eco", "environment"]
    }

    for target, subs in mapping_targets.items():
        found = find(subs)
        if found:
            col_map[found] = target

    # apply rename
    if col_map:
        df = df.rename(columns=col_map)

    return df

# -----------------------
# Pipeline builder
# -----------------------
NUMERIC_FEATURES = ["Age", "Income", "FMCG_Spend", "Electronics_Spend", "Entertainment_Spend"]
CATEGORICAL_FEATURES = ["City_Tier", "Brand_Preference"]
# We'll also use the computed "Lifestyle_Score" (numeric) as extra numeric feature
ALL_FEATURES = NUMERIC_FEATURES + ["Lifestyle_Score"] + CATEGORICAL_FEATURES

def build_pipeline(n_pca=2, k=4, random_state=42):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + ["Lifestyle_Score"]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("kmeans", KMeans(n_clusters=k, random_state=random_state))
    ])
    return pipeline

# -----------------------
# Helpers: save/load/train/predict
# -----------------------
def save_model(pipeline, path=MODEL_PATH):
    joblib.dump(pipeline, path)

def load_model(path=MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)

def train_and_save(df, k=4, n_pca=2):
    pipeline = build_pipeline(n_pca=n_pca, k=k)
    pipeline.fit(df[ALL_FEATURES])
    save_model(pipeline)
    return pipeline

def predict_df(df, pipeline):
    df2 = df.copy().reset_index(drop=True)
    # get processed features & predictions
    transformed = pipeline.named_steps["preproc"].transform(df2[NUMERIC_FEATURES + ["Lifestyle_Score"] + CATEGORICAL_FEATURES])
    pcs = pipeline.named_steps["pca"].transform(transformed)
    labels = pipeline.named_steps["kmeans"].predict(transformed)
    df2["pc1"] = pcs[:, 0]
    df2["pc2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0
    df2["Cluster"] = labels
    return df2

# -----------------------
# UI: Sidebar controls
# -----------------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"])
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
n_pca = st.sidebar.slider("PCA components", 2, 3, 2)
if st.sidebar.button("Delete saved model"):
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        st.sidebar.success("Deleted saved model.")
    else:
        st.sidebar.info("No saved model found.")

# -----------------------
# Main: Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(["Train", "Visualize", "Predict"])

# -----------------------
# Tab 1: Train
# -----------------------
with tab1:
    st.header("Train segmentation model")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Uploaded columns:", list(df.columns))
        df = smart_map = smart_map = smart_map = None  # keep flake happy (no-op)
        # map automatically with smart_map_columns
        df = smart_map_columns(df)
        st.write("Mapped columns:", list(df.columns))
    else:
        st.info("No CSV uploaded — you can upload a CSV with your dataset or use a generated sample.")
        if st.button("Generate sample dataset (500 rows)"):
            # create a sample using heuristics similar to your data
            rng = np.random.default_rng(42)
            df = pd.DataFrame({
                "Customer_ID": [f"C{1000+i}" for i in range(500)],
                "Age": rng.integers(18, 65, size=500),
                "Income": (rng.normal(60000, 25000, size=500)).clip(10000, 250000).astype(int),
                "City_Tier": np.random.choice(["Tier-1","Tier-2","Tier-3"], size=500, p=[0.25,0.45,0.30]),
                "FMCG_Spend": (rng.normal(1500,800,size=500)).clip(100,10000).round(1),
                "Electronics_Spend": (rng.normal(4000,6000,size=500)).clip(0,50000).round(1),
                "Entertainment_Spend": (rng.normal(800,900,size=500)).clip(0,15000).round(1),
                "Brand_Preference": np.random.choice(["Brand-A","Brand-B","Brand-C","Brand-D"], size=500),
                "fitness": np.random.choice([0,1], size=500, p=[0.65,0.35]),
                "family_oriented": np.random.choice([0,1], size=500, p=[0.55,0.45]),
                "tech_savvy": np.random.choice([0,1], size=500, p=[0.7,0.3]),
                "eco_friendly": np.random.choice([0,1], size=500, p=[0.8,0.2]),
            })
            st.success("Sample dataset generated.")
    # at this point df should exist
    if 'df' in locals():
        # standardize column names
        df = smart_map_columns(df)
        # ensure lower-case flag columns mapped properly
        # compute lifestyle score using Option B
        # find fitness/family/tech/eco columns in dataframe (they might be named differently)
        # define helper to find col by keyword
        def find_col(df, keywords):
            for kw in keywords:
                for c in df.columns:
                    if kw in c.lower():
                        return c
            return None

        fitness_col = find_col(df, ["fitness", "fit"])
        family_col = find_col(df, ["family", "family_oriented"])
        tech_col = find_col(df, ["tech_savvy", "tech", "tech_savv"])
        eco_col = find_col(df, ["eco_friendly", "eco", "environment"])

        # if any of these missing, set to 0
        for col_name in [fitness_col, family_col, tech_col, eco_col]:
            if col_name is None:
                # create missing column as zeros
                # add to df and show info
                # don't overwrite existing names
                pass

        # Create normalized columns if original names exist (map to standard names)
        if fitness_col and fitness_col not in df.columns:
            pass

        # for safety, create or coerce the expected binary columns
        if fitness_col is None:
            df["fitness"] = 0
            fitness_col = "fitness"
        if family_col is None:
            df["family_oriented"] = 0
            family_col = "family_oriented"
        if tech_col is None:
            df["tech_savvy"] = 0
            tech_col = "tech_savvy"
        if eco_col is None:
            df["eco_friendly"] = 0
            eco_col = "eco_friendly"

        # compute Lifestyle_Score
        # coerce to numeric (in case they are strings)
        df[fitness_col] = pd.to_numeric(df[fitness_col], errors='coerce').fillna(0).astype(int)
        df[family_col] = pd.to_numeric(df[family_col], errors='coerce').fillna(0).astype(int)
        df[tech_col] = pd.to_numeric(df[tech_col], errors='coerce').fillna(0).astype(int)
        df[eco_col] = pd.to_numeric(df[eco_col], errors='coerce').fillna(0).astype(int)

        df["Lifestyle_Score"] = df[fitness_col] + df[family_col] + df[tech_col] + df[eco_col]

        st.subheader("Preview (after mapping and Lifestyle_Score)")
        st.dataframe(df.head(6))

        # ensure required features exist (NUMERIC + CATEGORICAL)
        missing_req = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
        if missing_req:
            st.error(f"Missing required columns for pipeline: {missing_req}")
        else:
            st.info("Ready to train. Confirm settings and click Train.")
            if st.button("Train model now"):
                try:
                    pipeline = build_pipeline(n_pca=n_pca, k=k)
                    pipeline.fit(df[NUMERIC_FEATURES + ["Lifestyle_Score"] + CATEGORICAL_FEATURES])
                    save_model = joblib.dump(pipeline, MODEL_PATH)
                    st.success("Model trained and saved.")
                    # silhouette on preprocessed space
                    preproc = pipeline.named_steps["preproc"].transform(df[NUMERIC_FEATURES + ["Lifestyle_Score"] + CATEGORICAL_FEATURES])
                    labels = pipeline.named_steps["kmeans"].labels_
                    try:
                        sil = silhouette_score(preproc, labels)
                        st.metric("Silhouette (preprocessed)", f"{sil:.3f}")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Training failed: {e}")

# -----------------------
# Tab 2: Visualize
# -----------------------
with tab2:
    st.header("Visualize trained clusters")
    pipe = load_model()
    if pipe is None:
        st.info("No model found — train first.")
    else:
        uploaded_vis = st.file_uploader("Upload dataset to visualize clusters (CSV)", type=["csv"], key="viz")
        if uploaded_vis:
            dfv = pd.read_csv(uploaded_vis)
            dfv = smart_map_columns(dfv)
            # ensure binary columns present or create zeros
            for c in ["fitness","family_oriented","tech_savvy","eco_friendly"]:
                if c not in dfv.columns:
                    dfv[c] = 0
            # compute lifestyle score
            dfv["Lifestyle_Score"] = pd.to_numeric(dfv.get("fitness",0), errors='coerce').fillna(0).astype(int) + \
                                     pd.to_numeric(dfv.get("family_oriented",0), errors='coerce').fillna(0).astype(int) + \
                                     pd.to_numeric(dfv.get("tech_savvy",0), errors='coerce').fillna(0).astype(int) + \
                                     pd.to_numeric(dfv.get("eco_friendly",0), errors='coerce').fillna(0).astype(int)
            # check
            missing_req = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in dfv.columns]
            if missing_req:
                st.error(f"Missing required columns: {missing_req}")
            else:
                # prepare transformed features
                transformed = pipe.named_steps["preproc"].transform(dfv[NUMERIC_FEATURES + ["Lifestyle_Score"] + CATEGORICAL_FEATURES])
                pcs = pipe.named_steps["pca"].transform(transformed)
                labs = pipe.named_steps["kmeans"].predict(transformed)
                df_plot = pd.DataFrame({"pc1": pcs[:,0], "pc2": pcs[:,1], "cluster": labs})
                fig, ax = plt.subplots(figsize=(8,5))
                sc = ax.scatter(df_plot["pc1"], df_plot["pc2"], c=df_plot["cluster"], cmap="tab10", alpha=0.7)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA scatter (clusters)")
                st.pyplot(fig)
                st.subheader("Cluster counts")
                st.bar_chart(df_plot["cluster"].value_counts().sort_index())

# -----------------------
# Tab 3: Predict
# -----------------------
with tab3:
    st.header("Predict new customers")
    pipe = load_model()
    if pipe is None:
        st.info("No model found — train first.")
    else:
        up_pred = st.file_uploader("Upload CSV to predict clusters (batch)", type=["csv"], key="pred")
        if up_pred:
            dfx = pd.read_csv(up_pred)
            dfx = smart_map_columns(dfx)
            # ensure binary flags present
            for c in ["fitness","family_oriented","tech_savvy","eco_friendly"]:
                if c not in dfx.columns:
                    dfx[c] = 0
            dfx["Lifestyle_Score"] = pd.to_numeric(dfx.get("fitness",0), errors='coerce').fillna(0).astype(int) + \
                                     pd.to_numeric(dfx.get("family_oriented",0), errors='coerce').fillna(0).astype(int) + \
                                     pd.to_numeric(dfx.get("tech_savvy",0), errors='coerce').fillna(0).astype(int) + \
                                     pd.to_numeric(dfx.get("eco_friendly",0), errors='coerce').fillna(0).astype(int)
            missing_req = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in dfx.columns]
            if missing_req:
                st.error(f"CSV missing required columns: {missing_req}")
            else:
                transformed = pipe.named_steps["preproc"].transform(dfx[NUMERIC_FEATURES + ["Lifestyle_Score"] + CATEGORICAL_FEATURES])
                pcs = pipe.named_steps["pca"].transform(transformed)
                labs = pipe.named_steps["kmeans"].predict(transformed)
                dfx_out = dfx.copy()
                dfx_out["Cluster"] = labs
                st.dataframe(dfx_out.head(50))
                st.download_button("Download predictions", dfx_out.to_csv(index=False), "predictions.csv", "text/csv")

# End
st.markdown("---")
st.caption("App computes Lifestyle_Score = fitness + family_oriented + tech_savvy + eco_friendly (Option B). Adjust mapping heuristics above if your column names differ.")
