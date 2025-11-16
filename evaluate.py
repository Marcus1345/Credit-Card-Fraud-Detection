# src/evaluate.py
import pandas as pd
from features import make_features
from models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import joblib
def evaluate(model_path: str, processed_csv: str):
    # 1) Load data & model
    df = pd.read_csv(processed_csv)
    model = load_model(model_path)
    
    # 2) Tạo features (lưu ý: make_features nếu học scaler mới sẽ khác,
    # trong production bạn nên load scaler đã lưu để transform test chính xác)
    scaler = joblib.load("models/scaler.joblib")
    X, y, _ = make_features(df, scaler=scaler)

    # 3) chia train/test để có test set (hoặc lưu test từ train.py để consistent)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 4) dự đoán
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    
    # 5) metrics
    roc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", roc)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    # 6) PR-AUC
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)
    print("PR-AUC:", pr_auc)

if __name__ == "__main__":
    evaluate("models/fraud_model.joblib", "Data/RawDataSet.csv")
