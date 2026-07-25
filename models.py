
import joblib
import lightgbm as lgb


def build_model():
    model = lgb.LGBMClassifier(
        # core
        objective="binary",
        boosting_type="gbdt",
        n_estimators=800,
        learning_rate=0.03,

        # tree control - giảm complexity chống overfit
        num_leaves=48,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,

        # Không dùng class_weight vì đã SMOTE - tránh double-dipping
        # class_weight đã bỏ, is_unbalance=False

        # regularization - tăng mạnh chống overfit
        reg_alpha=0.5,
        reg_lambda=0.8,

        # speed + stability
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    return model


def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
