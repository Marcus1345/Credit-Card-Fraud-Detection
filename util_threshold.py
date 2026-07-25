# util_threshold.py
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    f1_score,
)


def find_best_threshold(y_true, y_pred_proba, method="f1"):
    """
    Tìm threshold tối ưu.

    Methods:
        'f1'     : Scan thresholds để maximize F1-score (default, tốt nhất cho imbalanced data)
        'pr'     : Dựa trên Precision-Recall curve, maximize F1 = 2*P*R/(P+R)
        'youden' : Maximize TPR - FPR (Youden's J statistic)
    """
    if method == "f1":
        return _threshold_f1(y_true, y_pred_proba)
    elif method == "pr":
        return _threshold_pr(y_true, y_pred_proba)
    elif method == "youden":
        return _threshold_youden(y_true, y_pred_proba)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'f1', 'pr', or 'youden'.")


def _threshold_f1(y_true, y_pred_proba):
    """Scan nhiều threshold, chọn threshold cho F1 cao nhất."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thr, best_f1 = 0.5, 0.0

    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr


def _threshold_pr(y_true, y_pred_proba):
    """Dựa trên Precision-Recall curve, maximize F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # precision và recall có len = len(thresholds) + 1, cắt bớt phần tử cuối
    precision = precision[:-1]
    recall = recall[:-1]

    # Tránh chia cho 0
    denom = precision + recall
    f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def _threshold_youden(y_true, y_pred_proba):
    """Maximize TPR - FPR (Youden's J statistic) - phương pháp cũ."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    best_idx = np.argmax(tpr - fpr)
    return thresholds[best_idx]
