# data_generation.py

import numpy as np
import pandas as pd

def generate_city_tier(n, rng):
    return rng.choice([1, 2, 3], size=n, p=[0.25, 0.45, 0.30])

def generate_age(n, rng):
    return rng.integers(18, 65, size=n)

def generate_income(n, city_tier, rng):
    base = np.where(city_tier == 1, 80000,
           np.where(city_tier == 2, 50000, 25000))
    noise = rng.normal(0, base * 0.25, size=n)
    return np.clip(base + noise, 15000, None).astype(int)

def generate_spending_scores(n, city_tier, rng):
    base = np.column_stack([
        np.where(city_tier == 1, rng.normal(0.75, 0.12, n), rng.normal(0.45, 0.15, n)),
        np.where(city_tier == 1, rng.normal(0.60, 0.15, n), rng.normal(0.40, 0.15, n)),
        np.where(city_tier == 1, rng.normal(0.55, 0.15, n), rng.normal(0.35, 0.12, n))
    ])
    scores = np.clip(base, 0, 1) * 100
    return np.round(scores, 1)

def generate_brand_preferences(n, brands, rng):
    raw = rng.random((n, brands))
    prefs = raw / raw.sum(axis=1, keepdims=True)
    top_brand = np.argmax(prefs, axis=1) + 1
    return top_brand

def generate_lifestyle(n, rng):
    probs = [0.35, 0.50, 0.30, 0.20]
    return (rng.random((n, len(probs))) < probs).astype(int)

def create_synthetic_dataset(n=3000, brands=5, seed=42):
    rng = np.random.default_rng(seed)

    city_tier = generate_city_tier(n, rng)
    age = generate_age(n, rng)
    income = generate_income(n, city_tier, rng)
    spending = generate_spending_scores(n, city_tier, rng)
    fmcg, electronics, entertainment = spending.T
    top_brand = generate_brand_preferences(n, brands, rng)
    lifestyle = generate_lifestyle(n, rng)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n+1)],
        "age": age,
        "income": income,
        "city_tier": city_tier,
        "spend_fmcg": fmcg,
        "spend_electronics": electronics,
        "spend_entertainment": entertainment,
        "pref_brand": top_brand,
        "fitness": lifestyle[:, 0],
        "family_oriented": lifestyle[:, 1],
        "tech_savvy": lifestyle[:, 2],
        "eco_friendly": lifestyle[:, 3]
    })

    return df

if __name__ == "__main__":
    df = create_synthetic_dataset(3000)
    df.to_csv("../data/synthetic_consumer_survey.csv", index=False)
    print("Saved data → ../data/synthetic_consumer_survey.csv")
