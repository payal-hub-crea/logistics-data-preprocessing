"""
Week 2 Task: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Author: Payal Shinde

Reference dataset: DataCo Smart Supply Chain (public, Kaggle / Mendeley Data).
This script SIMULATES a similar raw logistics dataset with injected quality problems,
then cleans and preprocesses it step by step.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------
# 1. DATA COLLECTION (SIMULATION)
# ---------------------------------------------------------------
def simulate_raw_data(n=1000, seed=42):
    np.random.seed(seed)
    raw = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="6h").astype(str),
        "shipping_mode": np.random.choice(
            ["Standard Class", "standard class", "First Class",
             "Second Class", "Same Day", None],
            n, p=[0.35, 0.05, 0.2, 0.2, 0.15, 0.05]),
        "region": np.random.choice(["West", "East", "South", "Central", " west "], n),
        "scheduled_days": np.random.choice([1, 2, 4, 5], n),
        "actual_days": np.random.normal(3.5, 1.5, n).round(1),
        "quantity": np.random.randint(1, 8, n).astype(float),
        "sales": np.random.normal(200, 60, n).round(2),
    })

    # Inject typical data quality problems
    raw.loc[np.random.choice(n, 40, replace=False), "actual_days"] = np.nan  # missing
    raw.loc[np.random.choice(n, 30, replace=False), "quantity"] = np.nan     # missing
    raw.loc[np.random.choice(n, 25, replace=False), "sales"] = np.nan        # missing
    raw.loc[np.random.choice(n, 8, replace=False), "actual_days"] = 45       # outliers
    raw.loc[np.random.choice(n, 6, replace=False), "sales"] = 5000           # outliers
    raw.loc[np.random.choice(n, 5, replace=False), "actual_days"] = -2       # impossible
    raw = pd.concat([raw, raw.sample(20, random_state=1)], ignore_index=True)  # duplicates
    return raw


# ---------------------------------------------------------------
# 2. DATA QUALITY ASSESSMENT
# ---------------------------------------------------------------
def profile(df, title="DATA PROFILE"):
    print(f"\n===== {title} =====")
    print("Shape:", df.shape)
    missing = df.isna().sum()
    print(pd.DataFrame({"missing": missing,
                        "percent": (missing / len(df) * 100).round(2)}))
    print("Duplicate rows:", df.duplicated().sum())
    print("Negative actual_days:", (df["actual_days"] < 0).sum())


# ---------------------------------------------------------------
# 3. OUTLIER HELPERS
# ---------------------------------------------------------------
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def report_outliers(df, cols):
    for col in cols:
        low, high = iqr_bounds(df[col])
        mask = (df[col] < low) | (df[col] > high)
        z = np.abs(stats.zscore(df[col]))
        print(f"{col}: IQR bounds=({low:.2f}, {high:.2f}), "
              f"IQR outliers={mask.sum()}, Z-score outliers(|z|>3)={(z > 3).sum()}")


def boxplot(df, cols, filename):
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    for ax, col in zip(axes, cols):
        df.boxplot(column=col, ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------------------------------------------
# 4. FULL PREPROCESSING PIPELINE
# ---------------------------------------------------------------
def preprocess_logistics(df):
    df = df.copy()

    # Step 1: remove duplicates
    df = df.drop_duplicates()

    # Step 2: fix data types and text formatting
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ["shipping_mode", "region"]:
        df[col] = df[col].str.strip().str.title()

    # Step 3: impossible values -> NaN (then imputed)
    df.loc[df["actual_days"] < 0, "actual_days"] = np.nan

    # Step 4: missing values
    df["shipping_mode"] = df["shipping_mode"].fillna("Unknown")
    for col in ["actual_days", "quantity", "sales"]:
        df[col] = df[col].fillna(df[col].median())   # median is robust to outliers

    # Step 5: outlier detection report + capping (winsorization)
    print("\n===== OUTLIERS (after imputation) =====")
    report_outliers(df, ["actual_days", "sales"])
    boxplot(df, ["actual_days", "sales"], "outliers_before.png")
    for col in ["actual_days", "sales"]:
        low, high = iqr_bounds(df[col])
        df[col + "_capped"] = df[col].clip(lower=low, upper=high)
    boxplot(df, ["actual_days_capped", "sales_capped"], "outliers_after.png")

    # Step 6: feature engineering
    df["delay_days"] = df["actual_days_capped"] - df["scheduled_days"]
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    df["order_month"] = df["order_date"].dt.month
    df["order_dayofweek"] = df["order_date"].dt.dayofweek

    # Step 7: one-hot encoding
    df = pd.get_dummies(df, columns=["shipping_mode", "region"], drop_first=True)

    # Step 8: normalization / scaling
    num_cols = ["actual_days_capped", "sales_capped", "quantity"]
    df[[c + "_minmax" for c in num_cols]] = MinMaxScaler().fit_transform(df[num_cols])
    df[[c + "_std" for c in num_cols]] = StandardScaler().fit_transform(df[num_cols])

    return df


# ---------------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------------
def validate(clean_df):
    assert clean_df.duplicated().sum() == 0, "Duplicates still present"
    assert clean_df[["actual_days", "quantity", "sales"]].isna().sum().sum() == 0, \
        "Missing values still present"
    assert (clean_df["actual_days_capped"] >= 0).all(), "Negative delivery days remain"
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    raw = simulate_raw_data()
    raw.to_csv("logistics_raw.csv", index=False)

    profile(raw, "RAW DATA")
    clean = preprocess_logistics(raw)
    profile(clean, "CLEAN DATA")

    validate(clean)
    clean.to_csv("logistics_clean.csv", index=False)
    print("\nSaved: logistics_raw.csv, logistics_clean.csv, outliers_before.png, outliers_after.png")
    print("Final shape:", clean.shape)
