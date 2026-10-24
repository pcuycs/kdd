import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from src.drift_simulator import DriftSimulator
from src.metrics import calculate_psi, evaluate_model_performance

# 1. Khởi tạo dữ liệu giả lập (hoặc Tải Give Me Some Credit / Default Card)
np.random.seed(42)
n_samples = 10000
X_dummy = pd.DataFrame({
    'RevolvingUtilization': np.random.exponential(scale=0.5, size=n_samples),
    'Age': np.random.normal(loc=50, scale=12, size=n_samples),
    'DebtRatio': np.random.uniform(low=0.1, high=3.0, size=n_samples),
    'MonthlyIncome': np.random.lognormal(mean=8.5, sigma=0.8, size=n_samples)
})
y_dummy = np.random.binomial(n=1, p=0.08, size=n_samples)  # 8% vỡ nợ (Imbalanced)[cite: 1]

# 2. Chia dữ liệu theo quy trình[cite: 1]
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.3, stratify=y_dummy, random_state=42)

# 3. Tiền xử lý Baseline (Imputation & Scaling)[cite: 1]
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# 4. Huấn luyện các mô hình Baseline & SOTA[cite: 1]
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', scale_pos_weight=11.5, random_state=42),
    'LightGBM': LGBMClassifier(scale_pos_weight=11.5, random_state=42, verbose=-1),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

print("=== 1. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP BASELINE (NO DRIFT) ===")
baseline_results = {}
for name, model in models.items():
    model.fit(X_train_imp, y_train)
    y_prob = model.predict_proba(X_test_imp)[:, 1]
    res = evaluate_model_performance(y_test, y_prob)
    baseline_results[name] = res
    print(f"[{name}] AUC-ROC: {res['AUC-ROC']} | F1: {res['F1-Score']} | Brier: {res['Brier-Score']}")

# 5. Mô phỏng Kịch bản Feature Drift (+20%) & Missingness Shift (20%)[cite: 1]
print("\n=== 2. TIÊM DRIFT VÀO TẬP TEST (Mô phỏng cú sốc dữ liệu) ===")
key_features = ['RevolvingUtilization', 'DebtRatio', 'MonthlyIncome']
X_test_drifted = DriftSimulator.inject_feature_drift(X_test, key_features, shift_pct=0.20)
X_test_drifted = DriftSimulator.inject_missingness_shift(X_test_drifted, key_features, target_missing_rate=0.20)

# Đo lường Feature Drift bằng PSI[cite: 1]
for col in key_features:
    psi_val = calculate_psi(X_train[col].values, X_test_drifted[col].values)
    print(f"-> Chỉ số PSI của biến [{col}]: {psi_val:.4f} " + ("⚠️ (Cảnh báo Drift)" if psi_val > 0.25 else ""))

# 6. Đánh giá suy giảm hiệu năng các mô hình dưới Drift[cite: 1]
X_test_drifted_imp = imputer.transform(X_test_drifted)
print("\n=== 3. KẾT QUẢ SUY GIẢM HIỆU SUẤT DƯỚI TÁC ĐỘNG CỦA DRIFT ===")
for name, model in models.items():
    y_prob_drift = model.predict_proba(X_test_drifted_imp)[:, 1]
    res_drift = evaluate_model_performance(y_test, y_prob_drift)
    
    # Tính mức suy giảm tương đối delta_M[cite: 1]
    auc_base = baseline_results[name]['AUC-ROC']
    auc_drift = res_drift['AUC-ROC']
    delta_auc = (auc_base - auc_drift) / auc_base * 100
    
    print(f"[{name}] AUC-ROC mới: {res_drift['AUC-ROC']} (Suy giảm {delta_auc:.2f}%) | F1 mới: {res_drift['F1-Score']}")