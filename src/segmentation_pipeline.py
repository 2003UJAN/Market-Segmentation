# segmentation_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

def build_pipeline(n_pca=2, k=4, random_state=42):

    numeric = ["age", "income", "spend_fmcg", "spend_electronics", "spend_entertainment"]
    categorical = ["city_tier", "pref_brand"]
    binary = ["fitness", "family_oriented", "tech_savvy", "eco_friendly"]

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
            ("bin", "passthrough", binary)
        ]
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("kmeans", KMeans(n_clusters=k, random_state=random_state))
    ])

    return model


def train_pipeline(df, k=4, n_pca=2, save_path="../models/segmentation_pipeline.joblib"):
    model = build_pipeline(n_pca=n_pca, k=k)
    model.fit(df)
    joblib.dump(model, save_path)
    return model


def load_pipeline(path="../models/segmentation_pipeline.joblib"):
    return joblib.load(path)


def predict_segments(df, model):
    transformed = model.named_steps["preprocess"].transform(df)
    pcs = model.named_steps["pca"].transform(transformed)
    clusters = model.named_steps["kmeans"].predict(transformed)

    res = df.copy()
    res["pc1"] = pcs[:, 0]
    res["pc2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0
    res["cluster"] = clusters
    return res


def profile_clusters(df, model):
    df_pred = predict_segments(df, model)
    
    profile = df_pred.groupby("cluster").agg({
        "age": "median",
        "income": "median",
        "spend_fmcg": "mean",
        "spend_electronics": "mean",
        "spend_entertainment": "mean",
        "city_tier": lambda x: x.mode().iloc[0],
        "pref_brand": lambda x: x.mode().iloc[0],
        "fitness": "mean",
        "family_oriented": "mean",
        "tech_savvy": "mean",
        "eco_friendly": "mean"
    })

    return profile.reset_index()


if __name__ == "__main__":
    df = pd.read_csv("../data/synthetic_consumer_survey.csv")
    model = train_pipeline(df)
    print("Model trained & saved.")
