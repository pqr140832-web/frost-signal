"""
颜料老化色差预测 - 最终优化版 (v7)
核心策略：物理驱动的时间序列外推

关键洞察：
1. dietaE = sqrt((L-L0)² + (a-a0)² + (b-b0)²) 是确定公式，不需猜
2. 测试集的每条数据都在训练集中有对应的时间序列（只是需要外推到更远时刻）
3. 颜料老化 dE(t) 遵循 sqrt(t) 扩散规律（短期）+ 最近趋势线性外推（修正）
4. 两者加权组合，在 forward-eval 上 R²=0.96，MAE=0.40

评分分析：
- baseline (RandomForest) ≈ 19分
- 迭代2 (XGBoost+Optuna) ≈ 34分
- v6 (XGB+LGBM+CB ensemble, OOF R²=0.92) = 29.8分
  → 高OOF R²≠高测试分，因OOF在已有时间点插值，测试需要外推！
- 本方案在外推任务上 R²=0.96，预期分数显著高于34

使用方法：
    python main_final.py
    
输出文件：
    predict_out.csv（和数据文件放同一目录）
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "baseline_and_data"
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
OUT_CSV   = BASE_DIR / "predict_out.csv"

TARGET = "dietaE"


# ===================== 核心：单序列预测方法 =====================

def predict_sqrt(t_train, dE_train, t_pred):
    """
    物理模型：dE ∝ sqrt(t)（扩散限制的老化行为）
    通过原点拟合：dE = A * sqrt(t)，OLS求解 A
    """
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 1:
        return 0.0
    ts = np.sqrt(t_train[mask])
    dm = dE_train[mask]
    A = np.dot(ts, dm) / np.dot(ts, ts)
    return float(max(A * np.sqrt(t_pred), 0))


def predict_last_rate(t_train, dE_train, t_pred):
    """
    趋势外推：用最后两个时间点的斜率线性外推
    更能捕捉近期的加速/减速趋势
    """
    if len(t_train) < 2:
        if len(t_train) == 1 and t_train[0] > 0:
            return float(max(dE_train[0] / t_train[0] * t_pred, 0))
        return 0.0
    t1, t2 = t_train[-2], t_train[-1]
    d1, d2 = dE_train[-2], dE_train[-1]
    if t2 == t1:
        return float(max(d2, 0))
    rate = (d2 - d1) / (t2 - t1)
    return float(max(d2 + rate * (t_pred - t2), 0))


def predict_powerlaw(t_train, dE_train, t_pred):
    """
    幂律模型：dE = A * t^alpha（在对数空间线性拟合）
    """
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 2:
        return predict_sqrt(t_train, dE_train, t_pred)
    log_t  = np.log(t_train[mask])
    log_dE = np.log(dE_train[mask])
    p = np.polyfit(log_t, log_dE, 1)
    return float(max(np.exp(p[1]) * (t_pred ** p[0]), 0))


def predict_single_series(t_train, dE_train, t_pred,
                           w_sqrt=0.30, w_last=0.70):
    """
    组合预测：sqrt模型 + 最近趋势外推
    权重经网格搜索优化（forward-eval MAE最低）:
        w_sqrt=0.30, w_last=0.70
    """
    p_sqrt = predict_sqrt(t_train, dE_train, t_pred)
    p_last = predict_last_rate(t_train, dE_train, t_pred)
    return float(max(w_sqrt * p_sqrt + w_last * p_last, 0))


# ===================== 主预测函数 =====================

def predict_testset(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
    """
    对每条测试记录：
    1. 在训练集中找到同一 (sample, aging_condition) 的时间序列
    2. 用组合时序模型外推到测试时刻
    3. 返回预测 dietaE 数组
    """
    results = []

    for _, row in df_test.iterrows():
        sample = row["sample"]
        cond   = row["aging_condition"]
        t_pred = float(row["aging_time_day"])

        sub = (
            df_train[
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond)
            ]
            .sort_values("aging_time_day")
        )

        if len(sub) == 0:
            # 找不到对应序列：用全局均值兜底
            results.append(float(df_train[TARGET].mean()))
            continue

        t_arr  = sub["aging_time_day"].values.astype(float)
        dE_arr = sub[TARGET].values.astype(float)

        pred = predict_single_series(t_arr, dE_arr, t_pred)
        results.append(pred)

    return np.array(results)


# ===================== 评估（在训练集上前向验证） =====================

def forward_eval(df_train: pd.DataFrame) -> dict:
    """
    评估策略：对每条时间序列，用"除最后一个时间点之外"的数据预测最后一个点。
    这模拟了测试集的外推场景，是比 OOF 更真实的验证方式。
    """
    y_true, y_pred = [], []

    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = (
                df_train[
                    (df_train["sample"] == sample) &
                    (df_train["aging_condition"] == cond)
                ]
                .sort_values("aging_time_day")
            )
            if len(sub) < 3:
                continue  # 需要至少 3 个点才能做有意义的外推评估

            train_sub = sub.iloc[:-1]
            test_row  = sub.iloc[-1]
            t_target  = float(test_row["aging_time_day"])

            t_arr  = train_sub["aging_time_day"].values.astype(float)
            dE_arr = train_sub[TARGET].values.astype(float)

            pred = predict_single_series(t_arr, dE_arr, t_target)
            y_true.append(float(test_row[TARGET]))
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "n":    len(y_true),
        "R2":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ===================== 主程序 =====================

def main():
    print("=" * 60)
    print("  颜料老化色差预测 - 物理时序外推版 (v7)")
    print("=" * 60)

    # 读数据
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test  = pd.read_csv(TEST_CSV,  encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行，测试集 {len(df_test)} 行")
    print(f"       训练样品数: {df_train['sample'].nunique()}")
    print(f"       测试样品数: {df_test['sample'].nunique()}")

    # 前向验证（模拟外推场景）
    print("\n[验证] 前向外推评估（每序列用前n-1点预测第n点）...")
    metrics = forward_eval(df_train)
    print(f"  有效序列数: {metrics['n']}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")

    # 测试集预测
    print("\n[预测] 生成测试集预测...")
    test_preds = predict_testset(df_train, df_test)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    # 保存
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[预测结果]")
    for i, (idx, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']} ({row['aging_condition']}, t={row['aging_time_day']}d): {test_preds[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"  最终指标 (forward-eval):")
    print(f"    R²   = {metrics['R2']:.4f}  (v6 OOF R²=0.92，但那是插值不是外推)")
    print(f"    MAE  = {metrics['MAE']:.4f}  (v6 OOF MAE 约 0.3，但测试实际更高)")
    print(f"    RMSE = {metrics['RMSE']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
