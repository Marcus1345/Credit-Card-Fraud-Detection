import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import numpy as np
import os
import time

from features import make_features
from models import build_model, save_model
from util_threshold import find_best_threshold


def run_model(csv_path, model_out, scaler_out, thr_out):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Making features...")
    X, y, scaler = make_features(df)

    joblib.dump(X.columns.tolist(), "models/features.joblib")

    print("Saving scaler...")
    joblib.dump(scaler, scaler_out)

    print("Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    print("\n" + "=" * 60)
    print("  DATA DISTRIBUTION - BEFORE RESAMPLE")
    print("=" * 60)
    train_counts = y_train.value_counts()
    print(f"  Normal (0): {train_counts.get(0, 0):>10,}")
    print(f"  Fraud  (1): {train_counts.get(1, 0):>10,}")
    print(f"  Ratio:      1 : {train_counts.get(0, 1) / max(train_counts.get(1, 1), 1):.0f}")

    print("\nHandling imbalance (SMOTE + UnderSample)...")

    sm = SMOTE(sampling_strategy=0.3, random_state=42)
    ru = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

    resampler = ImbPipeline([('smote', sm), ('under', ru)])
    X_res, y_res = resampler.fit_resample(X_train, y_train)

    print("\n" + "=" * 60)
    print("  DATA DISTRIBUTION - AFTER RESAMPLE")
    print("=" * 60)
    res_counts = pd.Series(y_res).value_counts()
    print(f"  Normal (0): {res_counts.get(0, 0):>10,}")
    print(f"  Fraud  (1): {res_counts.get(1, 0):>10,}")
    print(f"  Ratio:      1 : {res_counts.get(0, 1) / max(res_counts.get(1, 1), 1):.1f}")
    print(f"  Total samples: {len(y_res):,}")

    print("\nBuilding LightGBM model...")
    model = build_model()

    print("Training...")
    t0 = time.time()
    model.fit(X_res, y_res)
    print(f"Training done in {time.time()-t0:.2f} sec")

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Default Threshold = 0.50 ---")
    y_pred_default = (y_proba >= 0.5).astype(int)
    _print_evaluation(y_test, y_pred_default)

    print("\nFinding best threshold (F1-optimal)...")
    best_thr = find_best_threshold(y_test, y_proba, method="f1")
    print(f"Best threshold = {best_thr:.4f}")

    print(f"\n--- Optimized Threshold = {best_thr:.4f} ---")
    y_pred_opt = (y_proba >= best_thr).astype(int)
    _print_evaluation(y_test, y_pred_opt)

    pr_thr = find_best_threshold(y_test, y_proba, method="pr")
    print(f"\nPR-based threshold = {pr_thr:.4f}")
    y_pred_pr = (y_proba >= pr_thr).astype(int)
    print(f"  F1 = {f1_score(y_test, y_pred_pr):.5f}  "
          f"Precision = {precision_score(y_test, y_pred_pr):.5f}  "
          f"Recall = {recall_score(y_test, y_pred_pr):.5f}")

    print("\nSaving model + threshold...")
    save_model(model, model_out)
    open(thr_out, "w").write(str(best_thr))

    print("\nTRAINING PIPELINE COMPLETE!")
    print(f"Model saved: {model_out}")
    print(f"Threshold:   {best_thr:.4f}")


def _print_evaluation(y_test, y_pred):
    print("\n  CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"               Predicted 0    Predicted 1")
    print(f"  Actual 0:    {tn:>10,}    {fp:>10,}")
    print(f"  Actual 1:    {fn:>10,}    {tp:>10,}")

    print(f"\n  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    print("\n  CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, digits=5))

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"  Balanced Accuracy:  {bal_acc:.5f}")
    print(f"  MCC:                {mcc:.5f}")
    print(f"  F1 (fraud):         {f1:.5f}")
    print(f"  Precision (fraud):  {prec:.5f}")
    print(f"  Recall (fraud):     {rec:.5f}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    run_model(
        csv_path="Data/dataset_processed.csv",
        model_out="models/fraud_model.joblib",
        scaler_out="models/scaler.joblib",
        thr_out="models/best_threshold.txt"
    )
