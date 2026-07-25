"""
EDA.py - Exploratory Data Analysis & Data Preprocessing
========================================================
- Load raw data
- Handle null / NA / missing values
- Remove duplicate rows
- Drop unused columns (cc_num)
- Standardize & validate data
- Save cleaned dataset for model training
- Print comprehensive statistics
"""

import pandas as pd
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────
RAW_DATA_PATH = "Data/CleanData.csv"
PROCESSED_DATA_PATH = "Data/dataset_processed.csv"

TARGET_COL = "is_fraud"

# Columns to drop (identifiers, not useful for prediction)
DROP_COLS = ["cc_num"]

# ── 1. Load Data ──────────────────────────────────────────────
print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS & PREPROCESSING")
print("=" * 60)

if os.path.exists(PROCESSED_DATA_PATH):
    print(f"\n[1] Loading processed dataset: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
else:
    print(f"\n[1] Loading raw dataset: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Columns: {df.columns.tolist()}")

# ── 2. Basic Info ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  [2] DATA TYPES")
print(f"{'='*60}")
print(df.dtypes.to_string())

# ── 3. Descriptive Statistics ─────────────────────────────────
print(f"\n{'='*60}")
print("  [3] DESCRIPTIVE STATISTICS")
print(f"{'='*60}")
print(df.describe().to_string())

# ── 4. Handle Missing Values (null / NA / NaN) ───────────────
print(f"\n{'='*60}")
print("  [4] MISSING VALUES CHECK")
print(f"{'='*60}")

null_counts = df.isnull().sum()
na_total = null_counts.sum()

if na_total > 0:
    print(f"    Total missing values: {na_total:,}")
    print("    Per column:")
    for col, count in null_counts[null_counts > 0].items():
        pct = count / len(df) * 100
        print(f"      - {col}: {count:,} ({pct:.2f}%)")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"\n    Dropped {before - after:,} rows with null/NA values")
    print(f"    Remaining: {after:,} rows")
else:
    print("    No missing values found!")

# ── 5. Remove Duplicates ─────────────────────────────────────
print(f"\n{'='*60}")
print("  [5] DUPLICATE CHECK")
print(f"{'='*60}")

dup_count = df.duplicated().sum()
if dup_count > 0:
    print(f"    Found {dup_count:,} duplicate rows")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"    Removed {before - after:,} duplicates")
    print(f"    Remaining: {after:,} rows")
else:
    print("    No duplicate rows found!")

# ── 6. Drop Unused Columns ───────────────────────────────────
print(f"\n{'='*60}")
print("  [6] DROP UNUSED COLUMNS")
print(f"{'='*60}")

cols_to_drop = [c for c in DROP_COLS if c in df.columns]
if cols_to_drop:
    print(f"    Dropping: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    print(f"    Remaining columns ({len(df.columns)}): {df.columns.tolist()}")
else:
    print("    No unused columns to drop")

# ── 7. Validate Target Column ────────────────────────────────
print(f"\n{'='*60}")
print("  [7] TARGET VARIABLE ANALYSIS")
print(f"{'='*60}")

if TARGET_COL in df.columns:
    fraud_counts = df[TARGET_COL].value_counts()
    fraud_pct = df[TARGET_COL].value_counts(normalize=True) * 100

    print(f"    Target column: '{TARGET_COL}'")
    print(f"    Normal (0): {fraud_counts.get(0, 0):>10,}  ({fraud_pct.get(0, 0):.4f}%)")
    print(f"    Fraud  (1): {fraud_counts.get(1, 0):>10,}  ({fraud_pct.get(1, 0):.4f}%)")
    print(f"    Imbalance ratio: 1 : {fraud_counts.get(0, 1) / max(fraud_counts.get(1, 1), 1):.0f}")
else:
    print(f"    WARNING: Target column '{TARGET_COL}' not found!")
    print(f"    Available columns: {df.columns.tolist()}")

# ── 8. Feature Correlations with Target ──────────────────────
print(f"\n{'='*60}")
print("  [8] FEATURE CORRELATION WITH FRAUD")
print(f"{'='*60}")

if TARGET_COL in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numeric_cols:
        correlations = df[numeric_cols].corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
        print("    |  Feature         |  |Correlation|  |")
        print("    |-----------------|----------------|")
        for feat, corr in correlations.items():
            bar = "#" * int(corr * 50)
            print(f"    |  {feat:<15} |  {corr:.6f}       | {bar}")

# ── 9. Outlier Summary ───────────────────────────────────────
print(f"\n{'='*60}")
print("  [9] OUTLIER SUMMARY (IQR method)")
print(f"{'='*60}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        pct = outliers / len(df) * 100
        print(f"    {col:<15}: {outliers:>7,} outliers ({pct:.2f}%)")

# ── 10. Save Processed Dataset ───────────────────────────────
print(f"\n{'='*60}")
print("  [10] SAVE PROCESSED DATASET")
print(f"{'='*60}")

df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"    Saved to: {PROCESSED_DATA_PATH}")
print(f"    Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Final columns: {df.columns.tolist()}")

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"    Dataset ready for model training at: {PROCESSED_DATA_PATH}")
print(f"    Features: {len(df.columns) - 1}")
print(f"    Samples: {df.shape[0]:,}")
