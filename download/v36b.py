"""
颜料老化色差预测 - v36b
跨样本迁移学习 + LOLO/LOSO混合评估版

核心改进（vs v34）:
1. 统一组内池化建模: 同组所有样本数据池化拟合单一模型(而非每个样本单独拟合)
2. 个体偏差缩放: 用个体最后观测值与组中位数之比缩放组级预测
3. 通道级跨样本建模: dL/da/db分别用组内所有样本池化
4. 多模型加权平均: 组级模型用power law+log+linear+MM的最佳R2加权
5. 贝叶斯收缩: 对数据少的样本，预测更偏向组均值(减少过拟合)
6. LOLO+LOSO混合评估: 数据充足的组用LOLO，稀疏组(shu_red,other)用留一样本交叉验证
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error

SCRIPT_DIR = Path("/home/z/my-project/download")
DATA_DIR = Path("/home/z/my-project/upload/baseline_and_data")
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
TARGET = "dietaE"

TEST_WEIGHTS = {
    "dye": 5, "paper": 4, "shu_red": 14, "jade_green": 7, "cobalt_blue": 7, "other": 29
}
TOTAL_TEST = sum(TEST_WEIGHTS.values())


def detect_group(sample: str) -> str:
    if "翡翠绿" in sample: return "jade_green"
    if "钴蓝" in sample: return "cobalt_blue"
    if "曙红" in sample: return "shu_red"
    if "皮纸" in sample: return "paper"
    if any(x in sample for x in ["染料", "紫草", "苏木", "红花", "黄檗"]):
        return "dye"
    return "other"


def detect_subgroup(sample: str) -> str:
    """更细粒度的子分组"""
    if "矿物颜料" in sample: return "mineral"
    if "颜彩" in sample: return "gansai"
    if "中国画" in sample: return "chinese_painting"
    if "皮纸" in sample: return sample  # 每个皮纸单独
    if "曙红" in sample: return sample  # 每个曙红单独(但用组信息)
    if "翡翠绿" in sample: return sample
    if "钴蓝" in sample: return sample
    return sample


def prepare_series(df_sub):
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)


def prepare_channels(df_sub):
    agg = df_sub.groupby("aging_time_day").agg({
        "L": "mean", "a": "mean", "b": "mean",
        "L0": "first", "a0": "first", "b0": "first"
    }).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    t = agg["aging_time_day"].values.astype(float)
    dL = (agg["L"] - agg["L0"]).values.astype(float)
    da = (agg["a"] - agg["a0"]).values.astype(float)
    db = (agg["b"] - agg["b0"]).values.astype(float)
    return t, dL, da, db


def remove_outliers(t, y, threshold=2.5):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 3: return t.copy(), y.copy()
    diffs = np.diff(y)
    mean_abs = np.mean(np.abs(diffs))
    if mean_abs < 1e-6: return t, y
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t) - 1):
        if abs(diffs[i - 1]) > threshold * mean_abs:
            keep[i] = False
    return t[keep], y[keep]


# ===================== 生长模型库 =====================
def fit_power_law(t, y):
    mask = (t > 0) & (y > 0)
    if mask.sum() < 2: return None
    t, y = t[mask], y[mask]
    def neg_r2(n):
        tn = np.power(t, n)
        A = np.dot(tn, y) / (np.dot(tn, tn) + 1e-9)
        pred = A * tn
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return -(1 - ss_res / (ss_tot + 1e-9))
    res = minimize_scalar(neg_r2, bounds=(0.05, 3.0), method="bounded")
    best_n = res.x
    tn = np.power(t, best_n)
    A = np.dot(tn, y) / (np.dot(tn, tn) + 1e-9)
    return {"type": "power", "A": A, "n": best_n, "score": -res.fun}

def fit_linear(t, y):
    if len(t) < 2: return None
    A = np.vstack([np.ones_like(t), t]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "linear", "a": coeffs[0], "b": coeffs[1], "score": score}
    except: return None

def fit_log(t, y):
    mask = (t > 0) & (y > 0)
    if mask.sum() < 2: return None
    t, y = t[mask], y[mask]
    def neg_r2(logk):
        k = np.exp(logk)
        tk = np.log(1 + k * t)
        A = np.dot(tk, y) / (np.dot(tk, tk) + 1e-9)
        pred = A * tk
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return -(1 - ss_res / (ss_tot + 1e-9))
    res = minimize_scalar(neg_r2, bounds=(-5, 2), method="bounded")
    k = np.exp(res.x)
    tk = np.log(1 + k * t)
    A = np.dot(tk, y) / (np.dot(tk, tk) + 1e-9)
    return {"type": "log", "A": A, "k": k, "score": -res.fun}

def fit_mm(t, y):
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3: return None
    t, y = t[mask], y[mask]
    try:
        t_max, y_max = t.max(), y.max()
        def mm(tn, A, B): return A * tn / (B + tn)
        popt, _ = curve_fit(mm, t / t_max, y / y_max, p0=[1.0, 0.5], maxfev=5000)
        pred = y_max * mm(t / t_max, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "mm", "A": popt[0]*y_max, "B": popt[1]*t_max, "score": score}
    except: return None

def fit_sqrt(t, y):
    mask = (t > 0)
    if mask.sum() < 2: return None
    t, y = t[mask], y[mask]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(np.vstack([np.sqrt(t), np.ones_like(t)]).T, y, rcond=None)
        pred = coeffs[0]*np.sqrt(t) + coeffs[1]
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "sqrt", "A": coeffs[0], "B": coeffs[1], "score": score}
    except: return None

def fit_exp_decay(t, y):
    mask = (t > 0) & (y >= 0)
    if mask.sum() < 3: return None
    t, y = t[mask], y[mask]
    try:
        y_max = y.max() + 1e-6
        def expdec(tn, k): return 1 - np.exp(-k * tn)
        popt, _ = curve_fit(expdec, t / t.max(), y / y_max, p0=[1.0], bounds=(0.01, 10), maxfev=5000)
        k = popt[0]
        pred = y_max * expdec(t / t.max(), k)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "exp_decay", "A": y_max, "k": k / t.max(), "score": score}
    except: return None

def predict_model(model, t_pred):
    if model is None: return None
    if model["type"] == "power": return max(model["A"] * (t_pred ** model["n"]), 0)
    elif model["type"] == "linear": return max(model["a"] + model["b"] * t_pred, 0)
    elif model["type"] == "log": return model["A"] * np.log(1 + model["k"] * t_pred)
    elif model["type"] == "mm": return max(model["A"] * t_pred / (model["B"] + t_pred), 0)
    elif model["type"] == "sqrt": return max(model["A"] * np.sqrt(t_pred) + model["B"], 0)
    elif model["type"] == "exp_decay": return max(model["A"] * (1 - np.exp(-model["k"] * t_pred)), 0)
    return None

def best_model(t, y, min_score=-1.0):
    models = [fit_power_law(t, y), fit_log(t, y), fit_linear(t, y), fit_mm(t, y), fit_sqrt(t, y), fit_exp_decay(t, y)]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models: return None
    return max(models, key=lambda m: m["score"])


def ensemble_predict(models, t_pred):
    """多个模型按R2加权平均预测"""
    if not models:
        return None
    # 过滤有效模型
    valid = [(m, predict_model(m, t_pred)) for m in models if predict_model(m, t_pred) is not None]
    if not valid:
        return None
    # 按R2分数加权
    total_weight = sum(max(m["score"], 0.01) for m, _ in valid)
    if total_weight <= 0:
        return None
    pred = sum(w * p for (m, p), w in [(v, max(m["score"], 0.01)) for m, p in v for v in [(valid)]] 
               ) / total_weight if False else sum(max(m["score"], 0.01) * p for m, p in valid) / total_weight
    return max(pred, 0)


# ===================== 跨样本池化组级模型 =====================
class CrossSampleGroupModel:
    """核心改进: 组内所有样本池化建模"""
    
    def __init__(self, df_train):
        self.group_models = {}       # group_cond -> list of models (ensemble)
        self.group_channel_models = {}
        self.group_pooled_data = {}  # group_cond -> (times, all_dE_values)
        self.group_channel_pooled = {}
        self.sample_adjustments = {} # sample -> {cond -> ratio_at_last_t}
        self._build(df_train)
    
    def _build(self, df_train):
        groups = ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]
        
        for group in groups:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            
            for cond in ["UV", "humid-_heat"]:
                key = f"{group}_{cond}"
                
                # === 1. 池化所有样本的dE数据 ===
                all_times = []
                all_dE = []
                channel_data = {"dL": [], "da": [], "db": []}
                channel_times_list = []
                
                for sample in members:
                    sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
                    sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
                    if len(sub) == 0:
                        continue
                    
                    t_vals = sub["aging_time_day"].values.astype(float)
                    dE_vals = sub[TARGET].values.astype(float)
                    
                    # 对每个样本做归一化时间对齐(关键改进!)
                    # 用中位数dE序列作为组级"标准曲线"
                    for t, dE in zip(t_vals, dE_vals):
                        all_times.append(t)
                        all_dE.append(dE)
                    
                    # 通道数据
                    t_ch, dL, da, db = prepare_channels(sub)
                    for i, t in enumerate(t_ch):
                        channel_data["dL"].append(dL[i])
                        channel_data["da"].append(da[i])
                        channel_data["db"].append(db[i])
                        channel_times_list.append(t)
                    
                    # 记录个体在最后时间点的调整系数
                    last_t = t_vals[-1]
                    if sample not in self.sample_adjustments:
                        self.sample_adjustments[sample] = {}
                    self.sample_adjustments[sample][cond] = {
                        "last_t": last_t,
                        "last_dE": dE_vals[-1],
                    }
                
                if len(all_times) < 4:
                    continue
                
                times_arr = np.array(all_times)
                dE_arr = np.array(all_dE)
                
                # 保存池化数据
                self.group_pooled_data[key] = (times_arr, dE_arr)
                
                # === 2. 用中位数曲线建模(更稳健) ===
                unique_times = np.sort(np.unique(times_arr))
                median_dE = np.array([np.median(dE_arr[times_arr == t]) for t in unique_times])
                
                # 去除异常值后的中位数
                median_t, median_y = remove_outliers(unique_times, median_dE)
                
                if len(median_t) >= 2:
                    # 拟合多个模型
                    models = []
                    for fit_fn in [fit_power_law, fit_log, fit_linear, fit_mm, fit_sqrt, fit_exp_decay]:
                        m = fit_fn(median_t, median_y)
                        if m and m.get("score", -999) > 0:
                            models.append(m)
                    
                    if models:
                        self.group_models[key] = {
                            "models": models,
                            "median_t": median_t,
                            "median_y": median_y,
                            "unique_times": unique_times,
                            "median_dE_full": median_dE,
                        }
                
                # === 3. 通道级池化建模 ===
                if channel_times_list:
                    ch_times = np.array(channel_times_list)
                    ch_models = {}
                    for ch_name in ["dL", "da", "db"]:
                        ch_vals = np.array(channel_data[ch_name])
                        ch_unique_t = np.sort(np.unique(ch_times))
                        ch_median = np.array([np.median(ch_vals[ch_times == t]) for t in ch_unique_t])
                        
                        ch_model_list = []
                        for fit_fn in [fit_power_law, fit_log, fit_linear, fit_sqrt]:
                            m = fit_fn(ch_unique_t, np.abs(ch_median) + 1e-9)
                            if m and m.get("score", -999) > 0:
                                m["sign"] = np.sign(np.median(ch_median))
                                m["linear_raw"] = fit_linear(ch_unique_t, ch_median)
                                ch_model_list.append(m)
                        
                        if ch_model_list:
                            ch_models[ch_name] = {
                                "models": ch_model_list,
                                "times": ch_unique_t,
                                "medians": ch_median,
                            }
                    
                    if ch_models:
                        self.group_channel_models[key] = ch_models
                
                # === 4. 个体调整系数 ===
                if key in self.group_models:
                    gm = self.group_models[key]
                    med_t = gm["unique_times"]
                    med_dE = gm["median_dE_full"]
                    
                    for sample in members:
                        if cond in self.sample_adjustments[sample]:
                            adj = self.sample_adjustments[sample][cond]
                            last_t = adj["last_t"]
                            last_dE = adj["last_dE"]
                            
                            # 找组级中位数在最近时间点的值
                            closest_idx = np.argmin(np.abs(med_t - last_t))
                            group_dE_at_t = med_dE[closest_idx]
                            
                            if group_dE_at_t > 0.01:
                                adj["ratio"] = last_dE / group_dE_at_t
                            else:
                                adj["ratio"] = 1.0
                            
                            # 计算组级增长率(用于外推验证)
                            if len(med_t) >= 2:
                                sorted_pairs = sorted(zip(med_t, med_dE))
                                ratios = []
                                for i in range(1, len(sorted_pairs)):
                                    if sorted_pairs[i-1][1] > 0.01:
                                        ratios.append(sorted_pairs[i][1] / sorted_pairs[i-1][1])
                                if ratios:
                                    clean = [r for r in ratios if 0.5 < r < 5.0]
                                    adj["group_growth_ratios"] = clean
    
    def predict_group_ensemble(self, group, cond, t_pred):
        """用组级ensemble模型预测"""
        key = f"{group}_{cond}"
        if key not in self.group_models:
            return None
        
        gm = self.group_models[key]
        models = gm["models"]
        
        # 多模型加权预测
        pred = ensemble_predict(models, t_pred)
        return pred
    
    def predict_group_channel_ensemble(self, group, cond, t_pred):
        """用通道级ensemble预测"""
        key = f"{group}_{cond}"
        if key not in self.group_channel_models:
            return None
        
        ch_models = self.group_channel_models[key]
        ch_preds = {}
        
        for ch_name in ["dL", "da", "db"]:
            cm = ch_models[ch_name]
            models = cm["models"]
            
            # 用最好的单个模型预测
            best = max(models, key=lambda m: m.get("score", 0))
            abs_pred = predict_model(best, t_pred)
            
            # 也尝试线性模型
            if best.get("linear_raw"):
                lin_pred = predict_model(best["linear_raw"], t_pred)
                # 如果外推距离不大，线性更可靠
                max_t = cm["times"].max()
                if t_pred <= max_t * 1.5 and lin_pred is not None:
                    # 用线性
                    ch_preds[ch_name] = lin_pred
                elif abs_pred is not None:
                    ch_preds[ch_name] = best["sign"] * abs_pred
                else:
                    ch_preds[ch_name] = cm["medians"][-1]
            elif abs_pred is not None:
                ch_preds[ch_name] = best["sign"] * abs_pred
            else:
                ch_preds[ch_name] = cm["medians"][-1]
        
        if len(ch_preds) == 3:
            return float(max(np.sqrt(ch_preds["dL"]**2 + ch_preds["da"]**2 + ch_preds["db"]**2), 0))
        return None
    
    def predict_adjusted(self, sample, cond, t_pred, df_train):
        """个体调整后的预测: 组预测 * 个体偏差系数"""
        group = detect_group(sample)
        
        # 组级预测
        p_grp = self.predict_group_ensemble(group, cond, t_pred)
        p_ch = self.predict_group_channel_ensemble(group, cond, t_pred)
        
        # 获取个体调整信息
        adj = self.sample_adjustments.get(sample, {}).get(cond)
        
        result = {"grp_ensemble": p_grp, "ch_ensemble": p_ch}
        
        if adj and p_grp is not None:
            ratio = adj.get("ratio", 1.0)
            result["grp_adjusted"] = p_grp * ratio
        
        if adj and p_ch is not None:
            ratio = adj.get("ratio", 1.0)
            result["ch_adjusted"] = p_ch * ratio
        
        # 个体模型(如果有足够数据点)
        sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
        sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
        if len(sub) >= 2:
            t_ind, dE_ind = prepare_series(sub)
            t_c, dE_c = remove_outliers(t_ind, dE_ind)
            if len(t_c) >= 2:
                m = best_model(t_c, dE_c)
                if m:
                    result["individual"] = predict_model(m, t_pred)
                
                # 线性外推
                rate = (dE_c[-1] - dE_c[-2]) / (t_c[-1] - t_c[-2]) if t_c[-1] > t_c[-2] else 0
                result["linear_ext"] = max(dE_c[-1] + rate * (t_pred - t_c[-1]), 0)
                
                # 缩放组模型
                if p_grp is not None:
                    gmk = f"{group}_{cond}"
                    if gmk in self.group_models:
                        gm = self.group_models[gmk]
                        med_t = gm["unique_times"]
                        med_dE = gm["median_dE_full"]
                        closest_idx = np.argmin(np.abs(med_t - t_c[-1]))
                        gv = med_dE[closest_idx]
                        if gv > 0.01:
                            result["scaled_grp"] = p_grp * dE_c[-1] / gv
                
                # 比例外推
                if adj and "group_growth_ratios" in adj:
                    ratios = adj["group_growth_ratios"]
                    if ratios:
                        med_ratio = np.median(ratios)
                        gmk = f"{group}_{cond}"
                        if gmk in self.group_models:
                            gm = self.group_models[gmk]
                            med_t = gm["unique_times"]
                            steps = np.diff(med_t) if len(med_t) >= 2 else [1]
                            avg_step = np.mean(steps)
                            if avg_step > 0:
                                n_steps = (t_pred - t_c[-1]) / avg_step
                                result["ratio_ext"] = dE_c[-1] * (med_ratio ** n_steps)
        
        return result


# ===================== 10种预测策略 =====================
STRATEGY_NAMES = [
    "grp_ens", "ch_ens", "grp_adj", "ch_adj", "individual",
    "linear_ext", "scaled_grp", "ratio_ext", "bayesian_grp", "bayesian_ch"
]

def compute_all_strategies(csm, sample, cond, t_pred, df_train):
    """计算10种策略"""
    preds = csm.predict_adjusted(sample, cond, t_pred, df_train)
    if not preds:
        return None
    
    group = detect_group(sample)
    
    # 贝叶斯收缩策略: 根据个体数据量决定收缩强度
    sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
    sub = sub[sub["aging_time_day"] > 0]
    n_points = len(sub)
    
    # 收缩系数: 数据越少越偏向组均值
    # n_points=2: shrinkage=0.7 (70% group, 30% individual)
    # n_points=3: shrinkage=0.5
    # n_points=4: shrinkage=0.3
    # n_points>=5: shrinkage=0.1
    shrinkage = max(0.1, min(0.8, 1.0 - (n_points - 1) * 0.2))
    
    # 贝叶斯收缩: 组预测 + 个体调整的加权
    p_ind = preds.get("individual")
    p_grp = preds.get("grp_ens")
    p_grp_adj = preds.get("grp_adj")
    p_ch = preds.get("ch_ens")
    p_ch_adj = preds.get("ch_adj")
    
    bayesian_grp = None
    if p_grp is not None and p_grp_adj is not None:
        bayesian_grp = shrinkage * p_grp + (1 - shrinkage) * p_grp_adj
    elif p_grp is not None:
        bayesian_grp = p_grp
    
    bayesian_ch = None
    if p_ch is not None and p_ch_adj is not None:
        bayesian_ch = shrinkage * p_ch + (1 - shrinkage) * p_ch_adj
    elif p_ch is not None:
        bayesian_ch = p_ch
    
    return {
        "grp_ens": p_grp,
        "ch_ens": p_ch,
        "grp_adj": p_grp_adj,
        "ch_adj": p_ch_adj,
        "individual": p_ind,
        "linear_ext": preds.get("linear_ext"),
        "scaled_grp": preds.get("scaled_grp"),
        "ratio_ext": preds.get("ratio_ext"),
        "bayesian_grp": bayesian_grp,
        "bayesian_ch": bayesian_ch,
    }


def precompute_lolo_strategies(df_train):
    """LOLO + LOSO策略预计算
    LOLO: 对数据充足的组，leave-one-timepoint-out
    LOSO: 对数据稀疏的组(shu_red, other)，leave-one-sample-out
    """
    records = []
    
    # === LOLO: 对数据充足的组 ===
    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond)
            ].sort_values("aging_time_day")
            if len(sub) < 3: continue
            last_t = sub["aging_time_day"].max()
            train_df = df_train[~(
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond) &
                (df_train["aging_time_day"] == last_t)
            )]
            
            csm = CrossSampleGroupModel(train_df)
            t_pred = float(last_t)
            strats = compute_all_strategies(csm, sample, cond, t_pred, train_df)
            if strats is None: continue
            
            y_true = float(sub[sub["aging_time_day"] == last_t].iloc[-1][TARGET])
            record = {
                "sample": sample, "group": detect_group(sample), "cond": cond, "t": t_pred,
                "y_true": y_true,
            }
            for sn in STRATEGY_NAMES:
                record[sn] = strats[sn]
            records.append(record)
    
    # === LOSO: 对数据稀疏的组(shu_red, other) ===
    sparse_groups = ["shu_red", "other"]
    for group in sparse_groups:
        members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
        if len(members) < 3: continue  # 至少需要3个样本才能LOSO
        
        for cond in ["UV", "humid-_heat"]:
            for held_out_sample in members:
                # 用其余样本训练组级模型
                train_df = df_train[~(
                    (df_train["sample"] == held_out_sample) &
                    (df_train["aging_condition"] == cond)
                )]
                
                csm = CrossSampleGroupModel(train_df)
                
                # 对held-out样本的每个时间点评估
                held_sub = df_train[
                    (df_train["sample"] == held_out_sample) &
                    (df_train["aging_condition"] == cond)
                ].sort_values("aging_time_day")
                
                if len(held_sub) < 2: continue
                
                # 用第一个时间点后的所有点做评估
                # (用t=0和最早的非零点训练，预测后续点)
                non_zero = held_sub[held_sub["aging_time_day"] > 0]
                if len(non_zero) < 2: continue
                
                # 逐步评估: 用前N-1个点预测第N个点
                for i in range(1, len(non_zero)):
                    t_pred = float(non_zero.iloc[i]["aging_time_day"])
                    y_true = float(non_zero.iloc[i][TARGET])
                    
                    strats = compute_all_strategies(csm, held_out_sample, cond, t_pred, train_df)
                    if strats is None: continue
                    
                    record = {
                        "sample": held_out_sample, "group": group, "cond": cond, "t": t_pred,
                        "y_true": y_true,
                    }
                    for sn in STRATEGY_NAMES:
                        record[sn] = strats[sn]
                    records.append(record)
    
    return pd.DataFrame(records)


def eval_weights_weighted(lolo_df, weights_dict):
    """测试集加权MAE"""
    weighted_errors = []
    group_errors = {}
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = weights_dict.get(g, weights_dict.get("default", {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}))
        ws, wp = 0, 0
        for sn in STRATEGY_NAMES:
            v = row[sn]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ws += w.get(sn, 0)
                wp += w.get(sn, 0) * v
        if ws > 0:
            err = abs(row["y_true"] - wp / ws)
            tw = TEST_WEIGHTS.get(g, 1)
            weighted_errors.append(err * tw)
            if g not in group_errors: group_errors[g] = []
            group_errors[g].append(err)
    if not weighted_errors: return float("inf"), group_errors
    return sum(weighted_errors) / TOTAL_TEST, group_errors


def eval_weights_unweighted(lolo_df, weights_dict):
    """简单平均MAE"""
    all_errors = []
    group_errors = {}
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = weights_dict.get(g, weights_dict.get("default", {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}))
        ws, wp = 0, 0
        for sn in STRATEGY_NAMES:
            v = row[sn]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ws += w.get(sn, 0)
                wp += w.get(sn, 0) * v
        if ws > 0:
            err = abs(row["y_true"] - wp / ws)
            all_errors.append(err)
            if g not in group_errors: group_errors[g] = []
            group_errors[g].append(err)
    return np.mean(all_errors) if all_errors else float("inf"), group_errors


def search_best_weights(lolo_df, n_trials=30000, eval_fn=None):
    """搜索最优权重"""
    if eval_fn is None:
        eval_fn = eval_weights_weighted
    np.random.seed(42)
    groups = ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]
    best_weights = {}
    
    for group in groups:
        gdf = lolo_df[lolo_df["group"] == group]
        if len(gdf) < 2:
            best_weights[group] = {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}
            continue
        
        best_mae = float("inf")
        best_w = None
        
        for trial in range(n_trials):
            raw_w = np.random.dirichlet(np.ones(len(STRATEGY_NAMES)) * 0.3)
            mask = np.random.random(len(STRATEGY_NAMES)) > 0.15
            raw_w *= mask
            if raw_w.sum() < 0.01: continue
            raw_w /= raw_w.sum()
            
            w_dict = dict(zip(STRATEGY_NAMES, raw_w))
            mae, _ = eval_fn(gdf, {group: w_dict})
            if mae < best_mae:
                best_mae = mae
                best_w = raw_w.copy()
        
        if best_w is not None:
            best_weights[group] = dict(zip(STRATEGY_NAMES, best_w))
            uw_mae, _ = eval_weights_unweighted(gdf, {group: best_weights[group]})
            print(f"  {group:15s}: wMAE={best_mae:.4f}, uMAE={uw_mae:.4f}")
        else:
            best_weights[group] = {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}
    
    return best_weights


def make_prediction(csm, sample, cond, t_pred, df_train, weights):
    """用最优权重做预测"""
    strats = compute_all_strategies(csm, sample, cond, t_pred, df_train)
    if strats is None: return 0.0
    
    group = detect_group(sample)
    w = weights.get(group, {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES})
    ws, wp = 0, 0
    for sn in STRATEGY_NAMES:
        v = strats[sn]
        if v is not None:
            ws += w.get(sn, 0)
            wp += w.get(sn, 0) * v
    return wp / ws if ws > 0 else 0.0


def main():
    print("=" * 60)
    print("  颜料老化色差预测 v36b")
    print("  跨样本迁移学习 + 10策略 + 贝叶斯收缩 + LOLO/LOSO")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")
    
    # LOLO评估
    print("\n[LOLO] 预计算10策略...")
    lolo_df = precompute_lolo_strategies(df_train)
    print(f"  共 {len(lolo_df)} 个评估点")
    
    # 搜索最优权重
    print("\n[搜索] 最优权重(3万次/组)...")
    best_weights = search_best_weights(lolo_df, n_trials=30000)
    
    # 评估
    print("\n[评估] 最优权重:")
    opt_wmae, opt_ge = eval_weights_weighted(lolo_df, best_weights)
    opt_umae, _ = eval_weights_unweighted(lolo_df, best_weights)
    print(f"  测试集加权MAE = {opt_wmae:.4f}")
    print(f"  简单平均MAE = {opt_umae:.4f}")
    
    all_true, all_pred = [], []
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = best_weights.get(g, {sn: 1/10 for sn in STRATEGY_NAMES})
        ws, wp = 0, 0
        for sn in STRATEGY_NAMES:
            v = row[sn]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ws += w.get(sn, 0)
                wp += w.get(sn, 0) * v
        if ws > 0:
            all_true.append(row["y_true"])
            all_pred.append(wp / ws)
    
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    r2 = r2_score(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    print(f"  R2 = {r2:.4f}")
    print(f"  MAE = {mae:.4f}")
    
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        errs = opt_ge.get(g, [])
        if len(errs) >= 2:
            print(f"    {g:15s}: MAE={np.mean(errs):.4f}, n={len(errs)}")
    
    # 生成测试集预测
    print("\n[预测] 测试集...")
    csm = CrossSampleGroupModel(df_train)
    
    test_preds = []
    for _, row in df_test.iterrows():
        pred = make_prediction(csm, row["sample"], row["aging_condition"],
                               float(row["aging_time_day"]), df_train, best_weights)
        test_preds.append(float(max(pred, 0)))
    
    test_preds = np.array(test_preds)
    print(f"  范围: [{test_preds.min():.4f}, {test_preds.max():.4f}], 均值: {test_preds.mean():.4f}")
    
    # 保存
    pd.DataFrame({TARGET: test_preds}).to_csv(SCRIPT_DIR / "predict_out_v36.csv", index=False)
    pd.DataFrame({TARGET: test_preds}).to_csv(DATA_DIR / "predict_out.csv", index=False)
    
    print(f"\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d) -> {test_preds[i]:.4f}")
    
    # Ensemble with v34
    v34_path = SCRIPT_DIR / "predict_out_v34.csv"
    v14_path = SCRIPT_DIR / "predict_out_v14.csv"
    if v34_path.exists():
        v34 = pd.read_csv(v34_path)[TARGET].values
        for w36, label in [(0.3, "0.3v36_0.7v34"), (0.5, "0.5v36_0.5v34"), (0.7, "0.7v36_0.3v34")]:
            ens = w36 * test_preds + (1 - w36) * v34
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_v36_v34_{label}.csv", index=False)
        print(f"\n[v36+v34 ensemble] 已生成")
    
    if v14_path.exists():
        v14 = pd.read_csv(v14_path)[TARGET].values
        for w14, label in [(0.3, "0.3v14_0.7v36"), (0.5, "0.5v14_0.5v36")]:
            ens = w14 * v14 + (1 - w14) * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_v14_v36_{label}.csv", index=False)
    
    if v34_path.exists() and v14_path.exists():
        v34 = pd.read_csv(v34_path)[TARGET].values
        v14 = pd.read_csv(v14_path)[TARGET].values
        for w14, w34, label in [(0.2, 0.3, "0.2v14_0.3v34_0.5v36"), (0.3, 0.3, "0.3v14_0.3v34_0.4v36")]:
            ens = w14 * v14 + w34 * v34 + (1 - w14 - w34) * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_3ens_{label}.csv", index=False)
    
    print(f"\n[最优权重]")
    for g, w in best_weights.items():
        print(f"  {g:15s}: {w}")
    
    print(f"\nv36: R2={r2:.4f}, MAE={mae:.4f}, wMAE={opt_wmae:.4f}")
    print(f"v34: R2=0.967, MAE=0.340")


if __name__ == "__main__":
    main()
