import numpy as np
import pandas as pd

class DriftSimulator:
    """Mô phỏng các kịch bản Feature Drift và Missingness Shift"""
    
    @staticmethod
    def inject_feature_drift(df: pd.DataFrame, features: list, shift_pct: float) -> pd.DataFrame:
        """Kịch bản 1: Thay đổi trung bình/độ lệch chuẩn của đặc trưng"""
        df_drifted = df.copy()
        for col in features:
            if col in df_drifted.columns:
                # Dịch chuyển phân phối theo phần trăm
                std_val = df_drifted[col].std()
                df_drifted[col] = df_drifted[col] + (shift_pct * std_val)
        return df_drifted

    @staticmethod
    def inject_missingness_shift(df: pd.DataFrame, features: list, target_missing_rate: float) -> pd.DataFrame:
        """Kịch bản 2: Tăng tỷ lệ dữ liệu khuyết ngẫu nhiên (MCAR)"""
        df_drifted = df.copy()
        n_samples = len(df_drifted)
        for col in features:
            if col in df_drifted.columns:
                current_missing = df_drifted[col].isnull().mean()
                if target_missing_rate > current_missing:
                    additional_rate = target_missing_rate - current_missing
                    mask = np.random.rand(n_samples) < additional_rate
                    df_drifted.loc[mask, col] = np.nan
        return df_drifted