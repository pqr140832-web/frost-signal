"""
统一脚本：输入 (aging_time_day, L0, a0, b0)，输出预测 (dietaE)。

包含两部分：
1) 在 paint_aging_trainset.csv 中划分训练集/验证集并评估验证集指标
2) 在 paint_aging_testset.csv 上推理并保存预测结果
"""

# from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV_PATH = BASE_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV_PATH = BASE_DIR / "baseline_and_data" / "paint_aging_testset.csv"
PRED_OUTPUT_CSV = BASE_DIR / "baseline_and_data" / "predict_out.csv"

FEATURE_COLS = ["aging_time_day", "L0", "a0", "b0"]
TARGET_COLS = ["dietaE"]


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(TRAIN_CSV_PATH, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV_PATH, encoding="utf-8")

    df_train = df_train.dropna(axis=1, how="all")
    df_test = df_test.dropna(axis=1, how="all")

    missing_train = [c for c in (FEATURE_COLS + TARGET_COLS) if c not in df_train.columns]
    if missing_train:
        raise ValueError(f"[ERROR] Missing columns in train csv: {missing_train}")
    missing_test = [c for c in FEATURE_COLS if c not in df_test.columns]
    if missing_test:
        raise ValueError(f"[ERROR] Missing columns in test csv: {missing_test}")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COLS]
    X_test = df_test[FEATURE_COLS]
    return X_train, y_train, X_test


def split_train_valid(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """仅在训练数据中切分：80% train / 20% valid。"""
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def print_metrics(split_name: str, y_true: pd.DataFrame, y_pred: np.ndarray) -> None:
    mae_raw = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_raw = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2_raw = r2_score(y_true, y_pred, multioutput="raw_values")
    print(f"[{split_name}] Per-target metrics:")
    for name, mae_v, rmse_v, r2_v in zip(TARGET_COLS, mae_raw, rmse_raw, r2_raw):
        print(f"  {name}: MAE={mae_v:.4f}, RMSE={rmse_v:.4f}, R2={r2_v:.4f}")
    print(
        f"[{split_name}] Mean: MAE={np.mean(mae_raw):.4f}, "
        f"RMSE={np.mean(rmse_raw):.4f}, R2={np.mean(r2_raw):.4f}"
    )


def train_eval_and_predict_testset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """训练+验证评估，并对测试集推理。返回：y_valid, pred_valid, pred_test。"""
    X_all, y_all, X_test = load_train_test()
    X_train, X_valid, y_train, y_valid = split_train_valid(X_all, y_all)
    print(
        f"[SPLIT] train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}"
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    X_train_t = imputer.fit_transform(X_train)
    X_valid_t = imputer.transform(X_valid)
    X_test_t = imputer.transform(X_test)

    X_train_s = scaler.fit_transform(X_train_t)
    X_valid_s = scaler.transform(X_valid_t)
    X_test_s = scaler.transform(X_test_t)

    model.fit(X_train_s, y_train)
    pred_valid = model.predict(X_valid_s)
    pred_test = model.predict(X_test_s)
    return y_valid, pred_valid, pred_test


if __name__ == "__main__":
    y_valid, pred_valid, pred_test = train_eval_and_predict_testset()
    print_metrics("VALID", y_valid, pred_valid)

    if pred_test.ndim == 1:
        pred_test = pred_test.reshape(-1, 1)

    pred_df = pd.DataFrame(pred_test, columns=TARGET_COLS)
    pred_df.to_csv(PRED_OUTPUT_CSV, index=False)

    print(f"[OK] Saved prediction file: {PRED_OUTPUT_CSV}")
