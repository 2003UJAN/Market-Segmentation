import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

NUMERICAL_COLS = [
    "Age", "Income", "FMCG_Spend", "Electronics_Spend", "Entertainment_Spend"
]

CATEGORICAL_COLS = ["City_Tier", "Brand_Preference", "Lifestyle"]


def build_pipeline(n_pca=2, k=4):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("pca", PCA(n_components=n_pca)),
            ("kmeans", KMeans(n_clusters=k, random_state=42)),
        ]
    )
    return model


def train_pipeline(df, k=4, n_pca=2, save_path="models/segmentation_pipeline.joblib"):
    model = build_pipeline(k=k, n_pca=n_pca)
    model.fit(df)
    joblib.dump(model, save_path)
    return model


def load_pipeline(path):
    return joblib.load(path)
