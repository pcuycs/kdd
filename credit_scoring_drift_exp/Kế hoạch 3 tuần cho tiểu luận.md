“Feature Stability Analysis in Credit Risk Scoring: Temporal Evolution of Feature Importance and Its Impact on Model Robustness”

Hoặc ngắn hơn:

“Feature Importance Drift in Credit Risk Models: A Temporal Stability Study”

Tuần 1: Baseline & Data Preparation (5 ngày)
Ngày 1-2: Download Home Credit dataset, EDA → hiểu structure (case_id, week_num, target), tỷ lệ default 3.14%, missing %
Ngày 3-4: Reproduce CCI baseline (hoặc LightGBM đơn giản nếu code CCI chưa có)
Train trên weeks 1-40, test weeks 41-52
Tính baseline metrics: mean Gini, AUC-PR drop, stability metric
Ngày 5: Verify kết quả (Gini ~ 0.6, AUC-PR drop ~ 0.02-0.03)
Tuần 2: Feature Importance Tracking (6 ngày)
Ngày 6-7: Tính feature importance per-week (dùng SHAP values hoặc permutation importance từ LGB/XGB)
Output: ma trận (52 weeks × 80-100 features)
Ngày 8-9: Phân tích feature shift:
Rank feature importance từ week 1-20 vs week 41-52, tính rank correlation (Kendall τ)
Heatmap: feature importance evolution over time
Phát hiện: feature nào bị shift mạnh nhất? (ví dụ: feature X từ rank 5 → rank 30)
Ngày 10-11: So sánh:
Model A: Baseline CCI/LGB (dùng tất cả feature)
Model B: Baseline + feature importance tracking (metric: feature drift resistance)
Heatmap heatmap + correlation plot (feature importance shift vs model performance drop)
Tuần 3: Viết report & Finalize (4 ngày)
Ngày 12-13: Visualizations
Fig 1: Temporal feature importance heatmap
Fig 2: Top-k shifting features (rank change over time)
Fig 3: Correlation giữa feature importance shift và Gini drop
Ngày 14: Viết report (3-5 trang)
Mục tiêu chính: phát hiện feature importance không ổn định qua thời gian
Kết luận: features gặp shift → model phải thích nghi → suy giảm → cần dynamic handling
Ngày 15: Polish + submit