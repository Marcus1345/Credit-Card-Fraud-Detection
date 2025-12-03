# predict.py
import joblib
import numpy as np
import pandas as pd


def load_threshold(path="models/best_threshold.txt"):
    try:
        return float(open(path).read().strip())
    except:
        return 0.5   # fallback (không nên dùng)


def predict_single(model_path, scaler_path, thr_path, row_dict):
    """Dự đoán 1 dòng từ GUI"""

    # Load model + scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    threshold = load_threshold(thr_path)

    # Chuẩn đúng thứ tự cột
    cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

    # Chuyển dict → DataFrame
    df = pd.DataFrame([row_dict], columns=cols)

    # Scale Time & Amount
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    # Predict probability
    proba = model.predict_proba(df)[0, 1]

    # Apply threshold tối ưu
    label = 1 if proba >= threshold else 0

    return proba, label
