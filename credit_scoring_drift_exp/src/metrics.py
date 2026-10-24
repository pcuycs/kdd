import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, brier_score

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Tính Population Stability Index (PSI) để theo dõi Feature Drift"""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5
    
    expected_pct = np.histogram(expected, buckets)[0] / len(expected)
    actual_pct = np.histogram(actual, buckets)[0] / len(actual)
    
    # Tránh chia cho 0
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

def evaluate_model_performance(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Đánh giá toàn diện các chỉ số hiệu suất phân loại và hiệu chỉnh[cite: 1]"""
    y_pred = (y_prob >= threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "AUC-ROC": round(auc_roc, 4),
        "PR-AUC": round(pr_auc, 4),
        "F1-Score": round(f1, 4),
        "Brier-Score": round(brier, 4)
    }