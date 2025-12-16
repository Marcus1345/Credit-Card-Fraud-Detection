
import joblib
import lightgbm as lgb


def build_model():
    model = lgb.LGBMClassifier(
        # core
        objective="binary",
        boosting_type="gbdt",
        n_estimators=800,
        learning_rate=0.03,

        # tree control (chống overfit)
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,

        # fraud handling
        class_weight={0: 1, 1: 4},  # ưu tiên fraud
        is_unbalance=False,         # vì đã dùng SMOTE

        # regularization
        reg_alpha=0.5,
        reg_lambda=0.5,

        # speed + stability
        random_state=42,
        n_jobs=-1
    )

    return model


def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
