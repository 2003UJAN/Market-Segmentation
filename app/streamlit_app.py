import streamlit as st
import pandas as pd
from src.segmentation_pipeline import train_pipeline, load_trained_model

MODEL_PATH = "models/segmentation_model.pkl"

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
)

# -------------------------------------------------------
# CUSTOM CSS FOR SLEEK UI
# -------------------------------------------------------
st.markdown("""
<style>
/* Smooth rounded components */
div.stButton > button {
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-size: 16px;
}

/* Card-style containers */
.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: rgba(240,240,240,0.65);
    padding: 1.3rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

/* Dark mode friendly */
@media (prefers-color-scheme: dark) {
    .metric-card {
        background: rgba(20,20,20,0.4);
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.title("✨ Customer Segmentation Dashboard")
st.write("Upload data, train a segmentation model, and explore the customer clusters.")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

k_val = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=4)
pca_val = st.sidebar.slider("PCA Components", min_value=2, max_value=10, value=5)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV Dataset", type=["csv"])

train_btn = st.sidebar.button("🚀 Train Segmentation Model", use_container_width=True)

# -------------------------------------------------------
# MAIN AREA
# -------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    if train_btn:
        with st.spinner("⏳ Training model… please wait"):
            try:
                model = train_pipeline(df, k=k_val, n_pca=pca_val, save_path=MODEL_PATH)
                st.success("🎉 Model trained and saved successfully!")
            except Exception as e:
                st.error(f"Model training failed: {e}")

    # Load model section
    st.markdown("---")
    st.subheader("🔍 Predict Segments")

    model = load_trained_model(MODEL_PATH)
    if model is None:
        st.info("Upload data and train the model to enable predictions.")
    else:
        st.success("Model loaded successfully.")
        if st.button("Generate Customer Segments", use_container_width=True):
            preds = model.predict(df)
            df["Segment"] = preds

            st.dataframe(df)

            st.download_button(
                "📥 Download Segmented Data",
                df.to_csv(index=False),
                "segmented_output.csv",
                "text/csv",
                use_container_width=True
            )
else:
    st.info("Upload a dataset to begin.")
