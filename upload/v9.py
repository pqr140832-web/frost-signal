"""
颜料老化色差预测 - v9
核心变化（相对v7）：
1. 新增指数饱和模型作为长周期外推补充（权重≤0.15），避免皮纸等长周期高估
2. 新增有机染料短序列同类型先验约束，降低单样测量噪声
3. 极简差异化权重策略，仅区分"极短序列"和"长周期外推"两大场景，保持其他情况沿用v7的最优权重
4. 评估方式沿用v7的前向外推评估，确保验证的真实性和稳定性

使用方法：
    python main_final_v9.py
    
输出文件：
    predict_out_final_v9.csv（和数据文件放同一目录）
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== 路径配置（和v7一致）=====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "baseline_and_data"
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
OUT_CSV   = BASE_DIR / "predict_out_final_v9.csv"
TARGET = "dietaE"

# ===================== 1. 样品分类（用于先验约束）=====================
def get_sample_category(sample: str) -> str:
    sample_lower = sample.lower()
    if "皮纸" in sample:
        return "paper"
    elif "染料" in sample or "紫草" in sample or "苏木" in sample or "红花" in sample or "黄檗" in sample:
        return "organic_dye"
    else:
        return "general"

# ===================== 2. 核心模型定义（保留v7核心，新增2个补充）=====================
def predict_sqrt(t_train, dE_train, t_pred):
    """
    【v7核心】物理模型：dE ∝ sqrt(t)（扩散限制的老化行为）
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
    【v7核心】趋势外推：用最后两个时间点的斜率线性外推
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

def model_exp_sat(t, A, tau):
    """【v9新增】指数饱和模型：dE = A * (1 - exp(-t/tau))，仅用于长周期外推补充"""
    return A * (1 - np.exp(-t / tau))

def predict_exp_sat_robust(t_train, dE_train, t_pred):
    """【v9新增】稳健版指数饱和模型预测：带严格边界和失败兜底"""
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 3:
        return None
    t_fit = t_train[mask]
    dE_fit = dE_train[mask]
    try:
        p0 = [dE_fit.max() * 1.5, np.median(t_fit) * 2]
        bounds = ([dE_fit.max() * 0.5, 0.1], [dE_fit.max() * 5.0, max(t_fit) * 20.0])
        popt, _ = curve_fit(model_exp_sat, t_fit, dE_fit, p0=p0, bounds=bounds, maxfev=5000)
        return float(max(model_exp_sat(t_pred, *popt), 0))
    except:
        return None

def predict_sqrt_base(t_train, dE_train):
    """【v9新增】基础sqrt模型拟合：仅返回系数A，用于先验计算"""
    mask = (t_train > 0) & (dE_train > 0)
    if mask.sum() < 1:
        return None
    ts = np.sqrt(t_train[mask])
    dm = dE_train[mask]
    A = np.dot(ts, dm) / np.dot(ts, ts)
    return A

def predict_sqrt_with_prior(t_train, dE_train, t_pred, A_prior, prior_weight=0.3):
    """【v9新增】带先验约束的sqrt模型：仅用于有机染料短序列"""
    A_individual = predict_sqrt_base(t_train, dE_train)
    if A_individual is None:
        A_final = A_prior
    else:
        A_final = (1 - prior_weight) * A_individual + prior_weight * A_prior
    return float(max(A_final * np.sqrt(t_pred), 0))

# ===================== 3. 预计算：同类型先验系数 =====================
def compute_category_priors(df_train: pd.DataFrame) -> dict:
    """预计算有机染料类的平均sqrt系数A_prior"""
    priors = {}
    organic_dye_samples = df_train[df_train["sample"].apply(get_sample_category) == "organic_dye"]["sample"].unique()
    A_list = []
    for sample in organic_dye_samples:
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond)
            ].sort_values("aging_time_day")
            t_arr = sub["aging_time_day"].values.astype(float)
            dE_arr = sub[TARGET].values.astype(float)
            A = predict_sqrt_base(t_arr, dE_arr)
            if A is not None:
                A_list.append(A)
    if len(A_list) > 0:
        priors["organic_dye"] = np.mean(A_list)
    return priors

# ===================== 4. 极简差异化权重策略 =====================
def get_robust_weights(seq_len: int, t_pred: float, max_t_train: float, use_prior: bool = False) -> tuple:
    """
    【v9优化】极简权重：
    - 极短序列 (<=3个点)：纯sqrt模型
    - 长周期外推 (t_pred > 2*max_t_train)：sqrt(0.25) + 饱和(0.15) + 趋势(0.60)
    - 其他情况：沿用v7最优权重 (sqrt=0.3, last=0.7)
    """
    if seq_len <= 3:
        return 1.0, 0.0, 0.0
    elif t_pred > 2 * max_t_train:
        return 0.25, 0.15, 0.60
    else:
        return (0.35, 0.00, 0.65) if use_prior else (0.30, 0.00, 0.70)

# ===================== 5. 主预测函数 =====================
def predict_single_series_final(
    t_train, dE_train, t_pred, 
    sample_cat: str, 
    category_priors: dict
):
    """v9最终版单序列预测：整合v7核心 + v9优化"""
    seq_len = len(t_train)
    max_t_train = t_train.max() if seq_len > 0 else 0
    
    # 判断是否启用先验约束
    use_prior = False
    A_prior = None
    prior_weight = 0.0
    if sample_cat == "organic_dye" and seq_len <= 5 and sample_cat in category_priors:
        use_prior = True
        A_prior = category_priors[sample_cat]
        if seq_len <= 2:
            prior_weight = 0.5
        elif seq_len <= 4:
            prior_weight = 0.35
        else:
            prior_weight = 0.2
    
    # 获取权重
    w_sqrt, w_sat, w_last = get_robust_weights(seq_len, t_pred, max_t_train, use_prior)
    
    # 1. sqrt模型预测（带/不带先验）
    if use_prior:
        p_sqrt = predict_sqrt_with_prior(t_train, dE_train, t_pred, A_prior, prior_weight)
    else:
        p_sqrt = predict_sqrt(t_train, dE_train, t_pred)
    
    # 2. 最近趋势外推
    p_last = predict_last_rate(t_train, dE_train, t_pred)
    
    # 3. 指数饱和模型补充
    p_sat = 0.0
    if w_sat > 0:
        p_sat_candidate = predict_exp_sat_robust(t_train, dE_train, t_pred)
        if p_sat_candidate is not None:
            p_sat = p_sat_candidate
        else:
            w_sqrt += w_sat
            w_sat = 0.0
    
    # 组合预测
    final_pred = w_sqrt * p_sqrt + w_sat * p_sat + w_last * p_last
    return float(max(final_pred, 0))

def predict_testset_final(df_train: pd.DataFrame, df_test: pd.DataFrame, category_priors: dict) -> np.ndarray:
    """v9最终版测试集预测"""
    results = []
    for _, row in df_test.iterrows():
        sample = row["sample"]
        cond   = row["aging_condition"]
        t_pred = float(row["aging_time_day"])
        sample_cat = get_sample_category(sample)
        
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
        
        pred = predict_single_series_final(t_arr, dE_arr, t_pred, sample_cat, category_priors)
        results.append(pred)
    return np.array(results)

# ===================== 6. 评估（沿用v7的forward-eval）=====================
def forward_eval_final(df_train: pd.DataFrame, category_priors: dict) -> dict:
    """
    【v7核心】评估策略：对每条时间序列，用"除最后一个时间点之外"的数据预测最后一个点。
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
                continue
            train_sub = sub.iloc[:-1]
            test_row  = sub.iloc[-1]
            t_target  = float(test_row["aging_time_day"])
            t_arr  = train_sub["aging_time_day"].values.astype(float)
            dE_arr = train_sub[TARGET].values.astype(float)
            sample_cat = get_sample_category(sample)
            
            pred = predict_single_series_final(t_arr, dE_arr, t_target, sample_cat, category_priors)
            y_true.append(float(test_row[TARGET]))
            y_pred.append(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "n":    len(y_true),
        "R2":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("  颜料老化色差预测 - 最终优化版 (v9)")
    print("=" * 60)
    
    # 读数据
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test  = pd.read_csv(TEST_CSV,  encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行，测试集 {len(df_test)} 行")
    
    # 预计算先验
    category_priors = compute_category_priors(df_train)
    if "organic_dye" in category_priors:
        print(f"[先验] 有机染料类平均sqrt系数 A_prior = {category_priors['organic_dye']:.4f}")
    
    # 前向验证
    print("\n[验证] 前向外推评估（每序列用前n-1点预测第n点）...")
    metrics = forward_eval_final(df_train, category_priors)
    print(f"  有效序列数: {metrics['n']}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")
    
    # 测试集预测
    print("\n[预测] 生成最终版测试集预测...")
    test_preds = predict_testset_final(df_train, df_test, category_priors)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")
    
    # 保存
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")
    
    print("\n" + "=" * 60)
    print(f"  最终指标 (forward-eval):")
    print(f"    R²   = {metrics['R2']:.4f}  (v7: ≈0.96)")
    print(f"    MAE  = {metrics['MAE']:.4f}  (v7: ≈0.40)")
    print(f"    RMSE = {metrics['RMSE']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()