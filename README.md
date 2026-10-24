# Logistics Data Preprocessing Pipeline

Week 2 internship task: **Data Collection, Cleaning, and Preprocessing for Logistics Analysis**
Author: Payal Shinde

## Overview
This project simulates a real-world data preprocessing workflow for logistics (shipping and order fulfilment) data using Python and pandas. The reference dataset is the public **DataCo Smart Supply Chain** dataset (Kaggle / Mendeley Data). Because this is a simulation, a synthetic dataset with the same kind of structure and typical quality problems is generated, then cleaned step by step.

## Data quality problems simulated
| Problem | How it is handled |
|---|---|
| Duplicate records | `drop_duplicates()` |
| Inconsistent text (case, extra spaces) | `str.strip()` and `str.title()` |
| Wrong data types (dates as text) | `pd.to_datetime()` |
| Impossible values (negative delivery days) | Converted to NaN, then imputed |
| Missing values | Median for numeric, "Unknown" label for categorical |
| Outliers | Detected with IQR, Z-score and box plots; capped (winsorization) |
| Different scales | Min-Max normalization and Z-score standardization |

## Pipeline steps
1. Simulate data collection
2. Profile data quality (missing values, duplicates, invalid values)
3. Remove duplicates
4. Fix data types and text formatting
5. Handle impossible values and missing values
6. Detect and cap outliers
7. Feature engineering (`delay_days`, `is_late`, month, day of week)
8. One-hot encoding of categorical columns
9. Normalization and scaling
10. Validation checks on the cleaned data

## Project structure
```
logistics_preprocessing.py   # complete pipeline
requirements.txt             # dependencies
logistics_raw.csv            # simulated raw data
logistics_clean.csv          # cleaned, model-ready data
outliers_before.png          # box plots before capping
outliers_after.png           # box plots after capping
```

## How to run
```bash
pip install -r requirements.txt
python logistics_preprocessing.py
```

## Technologies
Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib

## Key learning
Data quality directly affects analytics and decisions in logistics: duplicates inflate demand, outliers distort average delivery time, and inconsistent labels hide true carrier performance. Cleaning and validating data first makes all later analysis more reliable.
