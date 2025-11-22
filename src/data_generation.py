import numpy as np
import pandas as pd

def create_synthetic_dataset(n=500):
    np.random.seed(42)

    data = {
        "Age": np.random.randint(18, 60, size=n),
        "Income": np.random.randint(20000, 150000, size=n),
        "City_Tier": np.random.choice(["Tier-1", "Tier-2", "Tier-3"], size=n),
        "FMCG_Spend": np.random.randint(500, 5000, size=n),
        "Electronics_Spend": np.random.randint(1000, 20000, size=n),
        "Entertainment_Spend": np.random.randint(200, 8000, size=n),
        "Brand_Preference": np.random.choice(["Brand-A", "Brand-B", "Brand-C"], size=n),
        "Lifestyle": np.random.choice(["Budget", "Moderate", "Premium"], size=n),
    }

    df = pd.DataFrame(data)
    return df
