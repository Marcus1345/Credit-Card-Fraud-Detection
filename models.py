import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Any

def build_model(n_estimators: int = 200, max_depth: int = 10, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        n_jobs=-1
    )
    return model

def save_model(model: Any, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
