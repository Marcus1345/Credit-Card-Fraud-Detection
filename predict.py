# src/predict.py
import pandas as pd
from models import load_model
import joblib
import numpy as np

# Danh sách các cột model đã train
FEATURE_COLUMNS = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

def predict_single(model_path: str, scaler_path: str, row: dict, threshold: float = 0.5):
    """
    row: dictionary của một transaction, keys có thể thiếu nhưng sẽ tự điền đủ các feature
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Điền các cột thiếu bằng 0.0
    for col in FEATURE_COLUMNS:
        if col not in row:
            row[col] = 0.0

    # Tạo DataFrame 1 hàng và sắp xếp đúng thứ tự cột
    X_new = pd.DataFrame([row])
    X_new = X_new[FEATURE_COLUMNS]

    # Scale Time & Amount
    X_new[['Time','Amount']] = scaler.transform(X_new[['Time','Amount']])

    # Predict
    proba = model.predict_proba(X_new)[:,1][0]  # xác suất fraud
    label = int(proba >= threshold)
    return proba, label


if __name__ == "__main__":
    # Ví dụ giả lập (không cần điền hết các V1-V28, sẽ tự điền 0)
    sample_row = {
        'Time': 100000,
        'V1': 0.1,
        'V2': -1.2,
        'Amount': 50.0
    }

    proba, label = predict_single(
        "models/fraud_model.joblib",
        "models/scaler.joblib",
        sample_row,
        threshold=0.3
    )

    print("Prob:", proba, "Label:", label)
