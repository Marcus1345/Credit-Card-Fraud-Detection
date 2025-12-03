
import joblib
import lightgbm as lgb

def build_model():
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=32,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
