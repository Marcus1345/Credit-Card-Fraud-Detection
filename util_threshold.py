import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    f1_score,
)


def find_best_threshold(y_true, y_pred_proba, method="f1"):
    if method == "f1":
        return _threshold_f1(y_true, y_pred_proba)
    elif method == "pr":
        return _threshold_pr(y_true, y_pred_proba)
    elif method == "youden":
        return _threshold_youden(y_true, y_pred_proba)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'f1', 'pr', or 'youden'.")


def _threshold_f1(y_true, y_pred_proba):
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
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def _threshold_youden(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    best_idx = np.argmax(tpr - fpr)
    return thresholds[best_idx]
