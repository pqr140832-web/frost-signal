import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

# ====================== 固定配置 ======================
TRAIN_CSV = Path("paint_aging_trainset.csv")
TEST_CSV = Path("paint_aging_testset.csv")
OUT_CSV = Path("pre.csv")
TARGET = "dietaE"

# ====================== 1. 样品分类 ======================
def get_sample_category(sample: str) -> str:
    sample_lower = sample.lower()
    if "皮纸" in sample:
        return "paper"
    elif "染料" in sample or "紫草" in sample or "苏木" in sample or "红花" in sample or "黄檗" in sample:
        return "organic_dye"
    else:
        return "general"

# ====================== 优化 2：异常点过滤 ======================
def remove_outliers(t, dE, threshold=2.0):
    if len(t) < 3:
        return t.copy(), dE.copy()
    t = np.asarray(t, dtype=float)
    dE = np.asarray(dE, dtype=float)
    diffs = np.diff(dE)
    if len(diffs) == 0:
        return t, dE
    mean_abs = np.mean(np.abs(diffs))
    if mean_abs < 1e-6:
        return t, dE
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)-1):
        if abs(diffs[i-1]) > threshold * mean_abs:
            keep[i] = False
    return t[keep], dE[keep]

# ====================== 基础模型 ======================
def predict_sqrt(t_train, dE_train, t_pred):
    t_train = np.asarray(t_train, dtype=float)
    dE_train = np.asarray(dE_train, dtype=float)
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 1:
        return 0.0
    ts = np.sqrt(t_train[mask])
    dm = dE_train[mask]
    A = np.dot(ts, dm) / (np.dot(ts, ts) + 1e-9)
    return float(max(A * np.sqrt(t_pred), 0))

def predict_last_rate(t_train, dE_train, t_pred):
    t_train = np.asarray(t_train, dtype=float)
    dE_train = np.asarray(dE_train, dtype=float)
    if len(t_train) < 2:
        if len(t_train) == 1 and t_train[0] > 0:
            return float(max(dE_train[0] / t_train[0] * t_pred, 0))
        return 0.0
    t1, t2 = t_train[-2], t_train[-1]
    d1, d2 = dE_train[-2], dE_train[-1]
    if t2 <= t1:
        return float(max(d2, 0))
    rate = (d2 - d1) / (t2 - t1)
    return float(max(d2 + rate * (t_pred - t2), 0))

def model_exp_sat(t, A, tau):
    return A * (1 - np.exp(-t / tau))

def predict_exp_sat_robust(t_train, dE_train, t_pred):
    t_train = np.asarray(t_train, dtype=float)
    dE_train = np.asarray(dE_train, dtype=float)
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 4:
        return None
    t_fit = t_train[mask]
    dE_fit = dE_train[mask]
    try:
        max_dE = dE_fit.max()
        p0 = [max_dE * 1.5, np.median(t_fit) + 1e-3]
        bounds = ([max_dE*0.5, 0.1], [max_dE*5.0, np.max(t_fit)*20])
        popt, _ = curve_fit(model_exp_sat, t_fit, dE_fit, p0=p0, bounds=bounds, maxfev=8000)
        return float(max(model_exp_sat(t_pred, *popt), 0))
    except:
        return None

def predict_sqrt_base(t_train, dE_train):
    t_train = np.asarray(t_train, dtype=float)
    dE_train = np.asarray(dE_train, dtype=float)
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 1:
        return None
    ts = np.sqrt(t_train[mask])
    dm = dE_train[mask]
    A = np.dot(ts, dm) / (np.dot(ts, ts) + 1e-9)
    return float(A)

def predict_sqrt_with_prior(t_train, dE_train, t_pred, A_prior, prior_weight=0.3):
    A_ind = predict_sqrt_base(t_train, dE_train)
    if A_ind is None:
        A_final = A_prior
    else:
        A_final = (1 - prior_weight) * A_ind + prior_weight * A_prior
    return float(max(A_final * np.sqrt(t_pred), 0))

# ====================== 优化 1：按【环境】计算先验 ======================
def compute_category_priors(df_train: pd.DataFrame) -> dict:
    priors = {}
    df_organic = df_train[df_train["sample"].apply(get_sample_category) == "organic_dye"]
    env_A = {}
    for cond in df_organic["aging_condition"].unique():
        sub = df_organic[df_organic["aging_condition"] == cond]
        A_list = []
        for sample in sub["sample"].unique():
            s = sub[sub["sample"] == sample].sort_values("aging_time_day")
            t = s["aging_time_day"].values.astype(float)
            dE = s[TARGET].values.astype(float)
            A = predict_sqrt_base(t, dE)
            if A is not None:
                A_list.append(A)
        if len(A_list) > 0:
            env_A[cond] = float(np.mean(A_list))
    priors["organic_dye"] = env_A
    return priors

# ====================== 优化 4：增强权重（饱和模型更早启用） ======================
def get_robust_weights(seq_len: int, t_pred: float, max_t_train: float, use_prior=False):
    if seq_len <= 2:
        return 1.0, 0.0, 0.0
    if seq_len >= 4:
        return 0.25, 0.25, 0.50
    if t_pred > 2 * max_t_train:
        return 0.25, 0.15, 0.60
    return (0.35, 0.0, 0.65) if use_prior else (0.30, 0.0, 0.70)

# ====================== 最终预测（整合所有优化） ======================
def predict_single_series_final(t_train, dE_train, t_pred, sample_cat: str, cond: str, category_priors: dict):
    seq_len = len(t_train)
    max_t = max(t_train) if seq_len > 0 else 0.0

    # 优化 2：过滤异常
    t_clean, dE_clean = remove_outliers(t_train, dE_train)
    seq_clean = len(t_clean)
    max_clean = max(t_clean) if seq_clean > 0 else 0.0

    # 优化 1：按环境取先验（完全修复类型错误）
    use_prior = False
    A_prior = None
    prior_weight = 0.0
    if sample_cat == "organic_dye" and seq_clean <= 5:
        cat_dict = category_priors.get("organic_dye", {})
        if cond in cat_dict:
            use_prior = True
            A_prior = float(cat_dict[cond])
            if seq_clean <= 2:
                prior_weight = 0.5
            elif seq_clean <= 4:
                prior_weight = 0.35
            else:
                prior_weight = 0.2

    # 优化 4：权重
    w1, w2, w3 = get_robust_weights(seq_clean, t_pred, max_clean, use_prior)

    # 三个模型
    if use_prior and A_prior is not None:
        p1 = predict_sqrt_with_prior(t_clean, dE_clean, t_pred, A_prior, prior_weight)
    else:
        p1 = predict_sqrt(t_clean, dE_clean, t_pred)

    p3 = predict_last_rate(t_clean, dE_clean, t_pred)
    p2 = 0.0
    if w2 > 0:
        val = predict_exp_sat_robust(t_clean, dE_clean, t_pred)
        if val is not None:
            p2 = val
        else:
            w1 += w2
            w2 = 0.0

    final = w1 * p1 + w2 * p2 + w3 * p3
    return float(max(final, 0.0))

# ====================== 测试集预测 ======================
def predict_testset_final(df_train: pd.DataFrame, df_test: pd.DataFrame, category_priors: dict) -> np.ndarray:
    res = []
    for _, row in df_test.iterrows():
        sample = row["sample"]
        cond = row["aging_condition"]
        t_pred = float(row["aging_time_day"])
        cat = get_sample_category(sample)
        sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
        sub = sub.sort_values("aging_time_day")
        if len(sub) == 0:
            res.append(float(df_train[TARGET].mean()))
            continue
        t = sub["aging_time_day"].values.astype(float)
        dE = sub[TARGET].values.astype(float)
        pred = predict_single_series_final(t, dE, t_pred, cat, cond, category_priors)
        res.append(pred)
    return np.array(res, dtype=float)

# ====================== 评估 ======================
def forward_eval_final(df_train: pd.DataFrame, category_priors: dict) -> dict:
    yt, yp = [], []
    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
            sub = sub.sort_values("aging_time_day")
            if len(sub) < 4:
                continue
            train = sub.iloc[:-1]
            test = sub.iloc[-1]
            t_t = float(test["aging_time_day"])
            t_arr = train["aging_time_day"].values.astype(float)
            dE_arr = train[TARGET].values.astype(float)
            cat = get_sample_category(sample)
            pred = predict_single_series_final(t_arr, dE_arr, t_t, cat, cond, category_priors)
            yt.append(float(test[TARGET]))
            yp.append(pred)
    yt = np.array(yt, dtype=float)
    yp = np.array(yp, dtype=float)
    return {
        "n": len(yt),
        "R2": r2_score(yt, yp),
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp)))
    }

# ====================== 主程序 ======================
def main():
    print("=" * 60)
    print(" 颜料老化预测 - 优化版 v9_plus ")
    print(" 已优化：环境先验 + 异常过滤 + 增强饱和模型 ")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"训练集 {len(df_train)} 行，测试集 {len(df_test)} 行")

    priors = compute_category_priors(df_train)
    print("先验已按环境计算完成")

    print("\n开始评估...")
    metrics = forward_eval_final(df_train, priors)
    print(f"有效样本：{metrics['n']}")
    print(f"R²  = {metrics['R2']:.4f}")
    print(f"MAE = {metrics['MAE']:.4f}")
    print(f"RMSE= {metrics['RMSE']:.4f}")

    print("\n开始预测...")
    preds = predict_testset_final(df_train, df_test, priors)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: preds}).to_csv(OUT_CSV, index=False)
    print(f"预测完成，已保存到：{OUT_CSV}")

if __name__ == "__main__":
    main()