import joblib
import pandas as pd


def load_threshold(path="models/best_threshold.txt"):
    try:
        return float(open(path).read().strip())
    except:
        return 0.5


def predict_single(model_path, scaler_path, thr_path, row_dict):
    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    threshold = load_threshold(thr_path)

    # 🔒 Load feature schema lúc train
    features = joblib.load("models/features.joblib")

    # Dict → DataFrame
    df = pd.DataFrame([row_dict])

    # Check thiếu feature
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu feature: {missing}")

    # ÉP đúng thứ tự + số lượng feature
    df = df[features]

    # Scale numeric columns (giống features.py)
    NUM_COLS = [
        "amt",
        "city_pop",
        "distance",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "unix_time"
    ]

    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    # Predict
    proba = model.predict_proba(df)[:, 1][0]
    label = int(proba >= threshold)

    return proba, label
