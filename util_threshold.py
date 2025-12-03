# util_threshold.py
import numpy as np
from sklearn.metrics import roc_curve

def find_best_threshold(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    best_idx = np.argmax(tpr - fpr)
    return thresholds[best_idx]
