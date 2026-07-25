"""
Credit Card Fraud Detection -- Flask Web Dashboard
API endpoints for prediction, model info, and evaluation.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from features import make_features, NUM_COLS
from util_threshold import find_best_threshold

# -- App Config --
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "dataset_processed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "best_threshold.txt")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.joblib")

# Cache for evaluation data
_eval_cache = {}


# ── Helpers ────────────────────────────────────────────────────

def load_model_artifacts():
    """Load model, scaler, threshold, features list."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = _load_threshold()
    features = joblib.load(FEATURES_PATH)
    return model, scaler, threshold, features


def _load_threshold():
    try:
        return float(open(THRESHOLD_PATH).read().strip())
    except Exception:
        return 0.5


def _predict_single(model, scaler, threshold, features, row_dict):
    """Predict a single transaction."""
    df = pd.DataFrame([row_dict])
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    df = df[features]
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    proba = model.predict_proba(df)[:, 1][0]
    label = int(proba >= threshold)
    return float(proba), int(label)


def _evaluate_model(model, X_test, y_test, features):
    """Evaluate model and return metrics dict."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, digits=5)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Subsample ROC for JSON efficiency
    step = max(1, len(fpr) // 200)
    fpr_sub = fpr[::step].tolist()
    tpr_sub = tpr[::step].tolist()

    # Feature importance
    fi = {}
    if hasattr(model, 'feature_importances_'):
        for fname, imp in zip(features, model.feature_importances_):
            fi[fname] = int(imp)

    best_thr = find_best_threshold(y_test, y_proba)

    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_data': {'fpr': fpr_sub, 'tpr': tpr_sub, 'auc': float(roc_auc)},
        'feature_importance': fi,
        'threshold': float(best_thr),
    }


# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# -- Predict --
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        row_dict = request.json
        model, scaler, threshold, features = load_model_artifacts()
        proba, label = _predict_single(model, scaler, threshold, features, row_dict)
        return jsonify({
            'proba': proba,
            'label': label,
            'threshold': threshold,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# -- Sample Generation --
# Cache for fraud/normal samples from dataset
_sample_cache = {'fraud': None, 'normal': None}


def _load_sample_pool():
    """Load and cache fraud/normal samples from dataset."""
    global _sample_cache
    if _sample_cache['fraud'] is not None:
        return True
    try:
        df = pd.read_csv(DATA_PATH)
        features = joblib.load(FEATURES_PATH)
        # Only keep feature columns that the model uses
        keep_cols = [c for c in features if c in df.columns]
        _sample_cache['fraud'] = df[df['is_fraud'] == 1][keep_cols]
        _sample_cache['normal'] = df[df['is_fraud'] == 0][keep_cols]
        return True
    except Exception:
        return False


def _generate_fraud_fallback():
    """Generate a fraud-like sample from real statistical distributions.
    Used only when dataset cannot be loaded."""
    # Based on actual dataset analysis:
    # amt: mean=531, median=396, P10=12, P90=1025
    # distance: mean=76, P10=36, P90=113 (max ~151, NOT 500-3000!)
    # hour: median=22 (late night is dominant for fraud)
    # category: mean=7.3, fraud skews toward higher categories
    # city_pop: median=2623, heavy right-skew
    hour = int(np.random.choice(
        [np.random.randint(22, 24), np.random.randint(0, 4),
         np.random.randint(0, 24)],
        p=[0.35, 0.25, 0.40]
    ))
    return {
        'amt': round(float(np.random.choice([
            np.random.uniform(200, 800),    # bulk of fraud (median ~396)
            np.random.uniform(800, 1200),   # high-value fraud
            np.random.uniform(10, 200),     # some low-value fraud exists
        ], p=[0.50, 0.30, 0.20])), 4),
        'city_pop': int(np.random.choice([
            np.random.randint(100, 5000),     # small city (most fraud)
            np.random.randint(5000, 100000),  # medium city
            np.random.randint(100000, 500000),  # larger city
        ], p=[0.55, 0.30, 0.15])),
        'lat': round(float(np.random.uniform(31, 45)), 4),
        'long': round(float(np.random.uniform(-112, -74)), 4),
        'merch_lat': round(float(np.random.uniform(31, 45)), 4),
        'merch_long': round(float(np.random.uniform(-112, -74)), 4),
        'unix_time': int(np.random.randint(1325466397, 1371787186)),
        'distance': round(float(np.random.uniform(35, 113)), 4),
        'merchant': int(np.random.randint(0, 693)),
        'category': int(np.random.choice(
            range(14), p=[0.03, 0.03, 0.05, 0.05, 0.05, 0.05,
                          0.08, 0.10, 0.12, 0.10, 0.10, 0.08, 0.10, 0.06]
        )),
        'hour': hour,
        'day': int(np.random.randint(1, 32)),
        'month': int(np.random.randint(1, 13)),
        'gender': int(np.random.randint(0, 2)),
    }


def _generate_normal_fallback():
    """Generate a normal-like sample from real statistical distributions."""
    return {
        'amt': round(float(np.random.choice([
            np.random.uniform(1, 50),       # small purchases (most common)
            np.random.uniform(50, 135),     # medium purchases
            np.random.uniform(1, 20),       # very small
        ], p=[0.50, 0.35, 0.15])), 4),
        'city_pop': int(np.random.choice([
            np.random.randint(200, 5000),
            np.random.randint(5000, 50000),
            np.random.randint(50000, 300000),
        ], p=[0.45, 0.35, 0.20])),
        'lat': round(float(np.random.uniform(31, 45)), 4),
        'long': round(float(np.random.uniform(-112, -74)), 4),
        'merch_lat': round(float(np.random.uniform(31, 45)), 4),
        'merch_long': round(float(np.random.uniform(-112, -74)), 4),
        'unix_time': int(np.random.randint(1325376018, 1371816817)),
        'distance': round(float(np.random.uniform(35, 113)), 4),
        'merchant': int(np.random.randint(0, 693)),
        'category': int(np.random.randint(0, 14)),
        'hour': int(np.random.randint(8, 20)),
        'day': int(np.random.randint(1, 32)),
        'month': int(np.random.randint(1, 13)),
        'gender': int(np.random.randint(0, 2)),
    }


@app.route('/api/sample/<sample_type>')
def api_sample(sample_type):
    """Generate sample transaction. Prefers real data, falls back to stats."""
    if sample_type not in ('fraud', 'normal'):
        return jsonify({'error': 'Invalid sample type'}), 400

    # Try to pick a real sample from dataset
    if _load_sample_pool():
        pool = _sample_cache[sample_type]
        if pool is not None and len(pool) > 0:
            row = pool.sample(1).iloc[0]
            sample = {col: (int(v) if isinstance(v, (np.integer,))
                            else round(float(v), 4))
                      for col, v in row.items()}
            return jsonify(sample)

    # Fallback: generate from statistical distributions
    if sample_type == 'fraud':
        sample = _generate_fraud_fallback()
    else:
        sample = _generate_normal_fallback()
    return jsonify(sample)


# -- Model Info --
@app.route('/api/model/info')
def api_model_info():
    try:
        model = joblib.load(MODEL_PATH)
        threshold = _load_threshold()
        features = joblib.load(FEATURES_PATH)

        params = model.get_params() if hasattr(model, 'get_params') else {}
        interesting_keys = [
            'n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
            'min_child_samples', 'subsample', 'colsample_bytree',
            'reg_alpha', 'reg_lambda', 'class_weight', 'boosting_type',
        ]
        filtered_params = {k: params[k] for k in interesting_keys if k in params}

        return jsonify({
            'model_type': type(model).__name__,
            'params': filtered_params,
            'threshold': threshold,
            'features': features,
            'num_features': len(features),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -- Evaluate Model --
@app.route('/api/model/evaluate')
def api_evaluate():
    global _eval_cache
    try:
        model_mtime = os.path.getmtime(MODEL_PATH)
        if _eval_cache.get('mtime') == model_mtime and _eval_cache.get('data'):
            return jsonify(_eval_cache['data'])

        df = pd.read_csv(DATA_PATH)
        X, y, scaler = make_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        features = X.columns.tolist()
        model = joblib.load(MODEL_PATH)

        eval_data = _evaluate_model(model, X_test, y_test, features)

        _eval_cache = {'mtime': model_mtime, 'data': eval_data}
        return jsonify(eval_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Main ───────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    print("\n[*] Fraud Detection Dashboard")
    print("    http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
