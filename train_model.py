# run_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import numpy as np
import os
import time

from features import make_features
from models import build_model, save_model
from util_threshold import find_best_threshold


def run_model(csv_path, model_out, scaler_out, thr_out):
    print(" Loading dataset...")
    df = pd.read_csv(csv_path)

    print(" Making features...")
    X, y, scaler = make_features(df)

    joblib.dump(X.columns.tolist(), "models/features.joblib")
    print("TRAIN FEATURES:", X.columns.tolist())


    print(" Saving scaler...")
    joblib.dump(scaler, scaler_out)

    print(" Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(" Handling imbalance (SMOTE + UnderSample)...")
    sm = SMOTE(random_state=42)
    ru = RandomUnderSampler(random_state=42)

    X_res, y_res = sm.fit_resample(X_train, y_train)
    X_res, y_res = ru.fit_resample(X_res, y_res)

    print(" Building LightGBM model...")
    model = build_model()

    print(" Training...")
    t0 = time.time()
    model.fit(X_res, y_res)
    print(f"Training done in {time.time()-t0:.2f} sec")

    print(" Evaluating...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=5))

    print(" Finding best threshold...")
    best_thr = find_best_threshold(y_test, y_proba)
    print(f"Best threshold = {best_thr:.4f}")

    print(" Saving model + threshold...")
    save_model(model, model_out)
    open(thr_out, "w").write(str(best_thr))

    print("\n TRAINING PIPELINE COMPLETE!")



if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    run_model(
        csv_path="Credit-Card-Fraud-Detection/Data/dataset_processed.csv",
        model_out="models/fraud_model.joblib",
        scaler_out="models/scaler.joblib",
        thr_out="models/best_threshold.txt"
    )
