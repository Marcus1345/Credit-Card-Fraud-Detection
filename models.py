import joblib
import lightgbm as lgb


def build_model():
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return model


def save_model(model, path):
    joblib.dump(model, path)
