import joblib
import pandas as pd


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


def load_threshold(path="models/best_threshold.txt"):
    try:
        return float(open(path).read().strip())
    except:
        return 0.5


def predict_single(model_path, scaler_path, thr_path, row_dict):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    threshold = load_threshold(thr_path)
    features = joblib.load("models/features.joblib")

    df = pd.DataFrame([row_dict])

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df = df[features]
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    proba = model.predict_proba(df)[:, 1][0]
    label = int(proba >= threshold)

    return proba, label
