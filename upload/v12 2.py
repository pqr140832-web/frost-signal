"""
颜料老化色差预测 - v12 (最终版)
基于 v11 的改进版

相对 v11 的改进：
1. 【核心】√t^n 模型替代固定 √t：每组使用搜索得到的最优幂次 n
   - 翡翠绿: n=0.45（比√t略慢，噪声大需要保守）
   - 钴蓝:   n=0.30（增长极慢，接近30次方根）
   - 曙红:   n=0.85（接近线性增长）
   - 皮纸:   n=0.50（保持√t，远外推最稳）
   - 染料:   n=0.35（慢于√t，避免近端过度放大）
   - 其他:   n=1.20（超线性，矿物颜料后期加速老化）

2. 权重微调（坐标下降搜索验证）：
   - 钴蓝: ws 0.55→0.75（更多依赖物理模型）
   - 曙红: ws 0.30→0.05（几乎完全依赖 last_rate）
   - 染料: ws 0.20→0.15
   - 其他: ws 0.30→0.05（last_rate 主导）

3. 保留 v11 的异常值过滤机制（已验证有效）

实测指标（forward-eval）：
  v11: MAE=0.3509, R²=0.9662, RMSE=0.4991
  v12: MAE=0.3244, R²=0.9693, RMSE=0.4761
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== 路径配置 =====================
# 自动识别脚本所在目录，在同目录下读取数据、输出结果
# 把本脚本和 train/test CSV 放同一个文件夹，直接运行即可
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = SCRIPT_DIR / "paint_aging_trainset.csv"
TEST_CSV  = SCRIPT_DIR / "paint_aging_testset.csv"
OUT_CSV   = SCRIPT_DIR / "predict_out.csv"
TARGET = "dietaE"

# ===================== 1. 样品分组 =====================
def detect_group(sample: str) -> str:
    if "翡翠绿" in sample: return "jade_green"
    if "钴蓝"  in sample: return "cobalt_blue"
    if "曙红"  in sample: return "shu_red"
    if "皮纸"  in sample: return "paper"
    if any(x in sample for x in ["染料", "紫草", "苏木", "红花", "黄檗"]):
        return "dye"
    return "other"

# ===================== 2. 搜索得到的最优参数 =====================
# 每组: (sqrt^n 的权重, 幂次 n)
# 通过坐标下降在 forward-eval 上搜索得到，3轮迭代收敛
GROUP_PARAMS = {
    "jade_green":  {"ws": 0.75, "n": 0.45},  # 噪声大，保守幂次，高物理模型权重
    "cobalt_blue": {"ws": 0.75, "n": 0.30},  # dE极小且波动，幂次很低
    "shu_red":     {"ws": 0.05, "n": 0.85},  # 接近线性增长，last_rate主导
    "paper":       {"ws": 0.55, "n": 0.50},  # 保持√t，远外推最稳
    "dye":         {"ws": 0.15, "n": 0.35},  # 慢幂次避免近端过度放大
    "other":       {"ws": 0.05, "n": 1.20},  # 矿物颜料超线性增长
}

# ===================== 3. 异常点过滤（v11 proven）=====================
def remove_outliers(t: np.ndarray, dE: np.ndarray, threshold: float = 2.0):
    t  = np.asarray(t,  dtype=float)
    dE = np.asarray(dE, dtype=float)
    if len(t) < 3:
        return t.copy(), dE.copy()
    diffs    = np.diff(dE)
    mean_abs = np.mean(np.abs(diffs))
    if mean_abs < 1e-6:
        return t, dE
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t) - 1):
        if abs(diffs[i - 1]) > threshold * mean_abs:
            keep[i] = False
    return t[keep], dE[keep]

# ===================== 4. 基础预测模型 =====================
def predict_t_power_n(t: np.ndarray, dE: np.ndarray, t_pred: float, n: float) -> float:
    """
    物理模型：dE = A · t^n
    OLS 强制过原点拟合 A，幂次 n 由组级参数固定。
    """
    mask = (t > 0) & (dE > 0)
    if mask.sum() < 1:
        return 0.0
    tn = np.power(t[mask], n)
    dm = dE[mask]
    A  = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
    return float(max(A * (t_pred ** n), 0.0))


def predict_last_rate(t: np.ndarray, dE: np.ndarray, t_pred: float) -> float:
    """
    趋势外推：用最近两个时间点的斜率线性外推。
    """
    if len(t) < 2:
        if len(t) == 1 and t[0] > 0:
            return float(max(dE[0] / t[0] * t_pred, 0.0))
        return 0.0
    t1, t2 = t[-2], t[-1]
    d1, d2 = dE[-2], dE[-1]
    if t2 <= t1:
        return float(max(d2, 0.0))
    rate = (d2 - d1) / (t2 - t1)
    return float(max(d2 + rate * (t_pred - t2), 0.0))

# ===================== 5. 核心预测函数 =====================
def predict_single_series(
    t_train:  np.ndarray,
    dE_train: np.ndarray,
    t_pred:   float,
    sample:   str,
) -> float:
    tc, dEc = remove_outliers(t_train, dE_train)
    group = detect_group(sample)
    params = GROUP_PARAMS[group]
    ws = params["ws"]
    n  = params["n"]

    ps = predict_t_power_n(tc, dEc, t_pred, n)
    pl = predict_last_rate(tc, dEc, t_pred)

    return float(max(ws * ps + (1.0 - ws) * pl, 0.0))

# ===================== 6. 测试集预测 =====================
def predict_testset(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
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
            results.append(float(df_train[TARGET].mean()))
            continue

        t_arr  = sub["aging_time_day"].values.astype(float)
        dE_arr = sub[TARGET].values.astype(float)

        pred = predict_single_series(t_arr, dE_arr, t_pred, sample)
        results.append(pred)

    return np.array(results, dtype=float)

# ===================== 7. 前向外推评估 =====================
def forward_eval(df_train: pd.DataFrame) -> dict:
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
                continue
            train_sub = sub.iloc[:-1]
            test_row  = sub.iloc[-1]
            t_arr  = train_sub["aging_time_day"].values.astype(float)
            dE_arr = train_sub[TARGET].values.astype(float)
            t_tgt  = float(test_row["aging_time_day"])
            pred = predict_single_series(t_arr, dE_arr, t_tgt, sample)
            y_true.append(float(test_row[TARGET]))
            y_pred.append(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "n":    len(y_true),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("  颜料老化色差预测 v12")
    print("  v11 + 组级最优幂次n + 坐标下降权重优化")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test  = pd.read_csv(TEST_CSV,  encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    print("\n[参数] 各组最优 (ws, n):")
    for g, p in GROUP_PARAMS.items():
        print(f"  {g:15s}: ws={p['ws']:.2f}, n={p['n']:.2f}")

    print("\n[评估] 前向外推评估...")
    metrics = forward_eval(df_train)
    print(f"  有效序列数: {metrics['n']}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")

    print("\n[预测] 生成测试集预测...")
    test_preds = predict_testset(df_train, df_test)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']} ({row['aging_condition']}, t={row['aging_time_day']}d)"
              f"  →  {test_preds[i]:.4f}")

    print()
    print("=" * 60)
    print(f"  v12:  R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    print(f"  v11:  R²=0.9662, MAE=0.3509, RMSE=0.4991")
    print(f"  v10:  R²=0.9587, MAE=0.4218, RMSE=0.5420")
    print("=" * 60)


if __name__ == "__main__":
    main()
