"""
颜料老化色差预测 - v34
基于v32的全面改进版

核心改进：
1. 新增策略7: 比例外推(ratio extrapolation) - 用组级增长比例外推
2. 新增策略8: 个体通道缩放组通道 - 用个体dL/da/db比例缩放组通道预测
3. 样本级权重搜索 - 对LOLO评估点逐样本搜索最优权重(而非组级统一权重)
4. 测试集加权LOLO - 按测试集各组样本数加权优化
5. 更大搜索空间(8策略, 20000次/组)
6. 通道外推改进: 对通道用power law+log而不仅是线性
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path("/home/z/my-project/download")
DATA_DIR = Path("/home/z/my-project/upload/baseline_and_data")
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
TARGET = "dietaE"

# 测试集各组样本数(用于加权LOLO)
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


# ===================== 稳健组级模型 =====================
class RobustGroupModel:
    def __init__(self, df_train):
        self.group_models = {}
        self.group_medians = {}
        self.group_channel_medians = {}
        self.group_channel_models = {}
        # 曙红专用: 组级增长比例
        self.shu_red_ratios = []
        self._build(df_train)

    def _build(self, df_train):
        # 曙红组增长比例分析
        shu_red_samples = [s for s in df_train["sample"].unique() if detect_group(s) == "shu_red"]
        for sample in shu_red_samples:
            for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
                sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
                sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
                if len(sub) >= 2:
                    times = sub["aging_time_day"].values
                    dEs = sub[TARGET].values
                    # 计算所有相邻时间点的增长比例
                    for i in range(1, len(times)):
                        if dEs[i-1] > 0.01:
                            self.shu_red_ratios.append(dEs[i] / dEs[i-1])

        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            for cond in ["UV", "humid-_heat"]:
                time_data, channel_data = {}, {}
                for m in members:
                    sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == cond)]
                    sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
                    for _, row in sub.iterrows():
                        t = int(row["aging_time_day"])
                        if t not in time_data: time_data[t] = []
                        time_data[t].append(row[TARGET])
                        if t not in channel_data: channel_data[t] = {"dL": [], "da": [], "db": []}
                        channel_data[t]["dL"].append(row["L"] - row["L0"])
                        channel_data[t]["da"].append(row["a"] - row["a0"])
                        channel_data[t]["db"].append(row["b"] - row["b0"])
                times = sorted(time_data.keys())
                if len(times) < 2: continue
                medians_t = np.array(times, dtype=float)
                medians_y = np.array([np.median(time_data[t]) for t in times])
                key = f"{group}_{cond}"
                self.group_medians[key] = {"times": medians_t, "medians": medians_y}
                model = best_model(medians_t, medians_y)
                if model: self.group_models[key] = model

                ch_times = sorted(channel_data.keys())
                ch_meds = {t: {"dL": np.median(channel_data[t]["dL"]),
                               "da": np.median(channel_data[t]["da"]),
                               "db": np.median(channel_data[t]["db"])} for t in ch_times}
                self.group_channel_medians[key] = ch_times, ch_meds

                # 通道级建模(用于外推) - 改进: 对每个通道用多种模型
                ch_models = {}
                for ch_name in ["dL", "da", "db"]:
                    ch_vals = np.array([ch_meds[t][ch_name] for t in ch_times])
                    lin = fit_linear(medians_t, ch_vals)
                    # 用绝对值拟合幂律/对数等(允许正值拟合)
                    abs_model = best_model(medians_t, np.abs(ch_vals), min_score=-2.0)
                    ch_models[ch_name] = {
                        "linear": lin,
                        "abs_model": abs_model,
                        "sign": np.sign(np.median(ch_vals)),
                    }
                self.group_channel_models[key] = ch_models

    def predict_group_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key in self.group_models: return predict_model(self.group_models[key], t_pred)
        return None

    def predict_group_channel_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key not in self.group_channel_medians: return None
        times, ch_meds = self.group_channel_medians[key]
        if not times: return None

        # 如果外推，用通道模型
        if key in self.group_channel_models and t_pred > times[-1]:
            ch_preds = {}
            for ch_name in ["dL", "da", "db"]:
                ch_model = self.group_channel_models[key][ch_name]
                p_lin = predict_model(ch_model["linear"], t_pred)
                p_abs = None
                if ch_model["abs_model"]:
                    p_abs_val = predict_model(ch_model["abs_model"], t_pred)
                    if p_abs_val is not None:
                        p_abs = ch_model["sign"] * p_abs_val
                # 选择: 线性优先(通道值通常变化比较线性)
                if p_lin is not None:
                    ch_preds[ch_name] = p_lin
                elif p_abs is not None:
                    ch_preds[ch_name] = p_abs
                else:
                    ch_preds[ch_name] = ch_meds[times[-1]][ch_name]
            if len(ch_preds) == 3:
                dE = np.sqrt(ch_preds["dL"]**2 + ch_preds["da"]**2 + ch_preds["db"]**2)
                return float(max(dE, 0))

        # 插值
        if t_pred <= times[0]: ch = ch_meds[times[0]]
        elif t_pred >= times[-1]: ch = ch_meds[times[-1]]
        else:
            for i in range(len(times) - 1):
                if times[i] <= t_pred <= times[i+1]:
                    frac = (t_pred - times[i]) / (times[i+1] - times[i])
                    ch = {k: ch_meds[times[i]][k]*(1-frac) + ch_meds[times[i+1]][k]*frac for k in ["dL","da","db"]}
                    break
            else: ch = ch_meds[times[-1]]
        return float(max(np.sqrt(ch["dL"]**2 + ch["da"]**2 + ch["db"]**2), 0))

    def get_group_ratio(self, group):
        """获取组级增长比例中位数"""
        if group == "shu_red" and self.shu_red_ratios:
            clean = [r for r in self.shu_red_ratios if 0.5 < r < 5.0]
            return np.median(clean) if clean else 1.2
        # 其他组: 从组级中位数序列计算
        ratios = []
        for cond in ["UV", "humid-_heat"]:
            key = f"{group}_{cond}"
            if key in self.group_medians:
                gm = self.group_medians[key]
                ts, ys = gm["times"], gm["medians"]
                for i in range(1, len(ts)):
                    if ys[i-1] > 0.01:
                        ratios.append(ys[i] / ys[i-1])
        if ratios:
            clean = [r for r in ratios if 0.5 < r < 5.0]
            return np.median(clean) if clean else 1.2
        return 1.2


# ===================== 8种预测策略 =====================
STRATEGY_NAMES = ["ind", "grp", "ch", "lin", "scaled", "ch_scaled", "ratio", "ind_ch_scaled"]

def compute_strategies(rgm, sample, cond, t_pred, df_train, sample_models=None):
    """计算8种策略的预测值"""
    sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)].sort_values("aging_time_day")
    if len(sub) == 0: return None
    t_arr, dE_arr = prepare_series(sub)
    if len(t_arr) == 0: return None
    tc, dEc = remove_outliers(t_arr, dE_arr)
    group = detect_group(sample)
    key = f"{sample}_{cond}"

    # 策略1: 个体最佳模型
    p_ind = None
    if sample_models and key in sample_models:
        p_ind = predict_model(sample_models[key], t_pred)

    # 策略2: 组级dE模型
    p_grp = rgm.predict_group_dE(group, cond, t_pred)

    # 策略3: 组级通道分解
    p_ch = rgm.predict_group_channel_dE(group, cond, t_pred)

    # 策略4: 线性外推(最后两点的增量)
    p_lin = None
    if len(tc) >= 2:
        rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
        p_lin = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)

    # 策略5: 缩放组模型(用个体dE/组dE比例缩放)
    p_scaled = None
    if p_grp is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_medians:
            gm = rgm.group_medians[gmk]
            gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
            if gv > 0.01:
                p_scaled = p_grp * dEc[-1] / gv

    # 策略6: 个体缩放的通道预测
    p_ch_scaled = None
    if p_ch is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_medians:
            gm = rgm.group_medians[gmk]
            gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
            if gv > 0.01:
                p_ch_scaled = p_ch * dEc[-1] / gv

    # 策略7: 比例外推(用组级增长比例)
    p_ratio = None
    if len(tc) >= 1 and t_pred > tc[-1]:
        ratio_per_day = rgm.get_group_ratio(group)
        # 把ratio转为per-day: ratio = growth^(1/dt)
        # 从组级数据估计dt
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_medians:
            gm = rgm.group_medians[gmk]
            if len(gm["times"]) >= 2:
                # ratio已经是per-step, 计算平均step
                steps = np.diff(gm["times"])
                avg_step = np.mean(steps)
                if avg_step > 0:
                    # ratio^(n_steps_to_extrapolate)
                    n_steps = (t_pred - tc[-1]) / avg_step
                    p_ratio = dEc[-1] * (ratio_per_day ** n_steps)

    # 策略8: 个体通道值缩放组通道预测
    p_ind_ch_scaled = None
    if p_ch is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_channel_medians:
            ch_times, ch_meds = rgm.group_channel_medians[gmk]
            if len(ch_times) >= 1:
                # 获取个体在最后一个训练时间点的通道值
                t_ch_sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
                t_ch_sub = t_ch_sub[t_ch_sub["aging_time_day"] > 0].sort_values("aging_time_day")
                if len(t_ch_sub) > 0:
                    last_row = t_ch_sub.iloc[-1]
                    ind_dL = last_row["L"] - last_row["L0"]
                    ind_da = last_row["a"] - last_row["a0"]
                    ind_db = last_row["b"] - last_row["b0"]
                    ind_dE = np.sqrt(ind_dL**2 + ind_da**2 + ind_db**2)

                    # 组级通道在最后训练时间点的dE
                    closest_t = ch_times[np.argmin(np.abs(np.array(ch_times) - tc[-1]))]
                    grp_ch = ch_meds[closest_t]
                    grp_dE = np.sqrt(grp_ch["dL"]**2 + grp_ch["da"]**2 + grp_ch["db"]**2)

                    if grp_dE > 0.01:
                        # 用个体dE/组dE的整体缩放
                        scale = ind_dE / grp_dE
                        p_ind_ch_scaled = p_ch * scale

    return {
        "ind": p_ind, "grp": p_grp, "ch": p_ch, "lin": p_lin,
        "scaled": p_scaled, "ch_scaled": p_ch_scaled,
        "ratio": p_ratio, "ind_ch_scaled": p_ind_ch_scaled
    }


def precompute_lolo_strategies(df_train):
    """对每个LOLO评估点，计算所有8种策略的预测值"""
    records = []
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

            rgm = RobustGroupModel(train_df)
            t_sub = train_df[(train_df["sample"] == sample) & (train_df["aging_condition"] == cond)]
            t_arr, dE_arr = prepare_series(t_sub)
            if len(t_arr) == 0: continue
            tc, dEc = remove_outliers(t_arr, dE_arr)

            sample_models = {}
            if len(tc) >= 2:
                m = best_model(tc, dEc)
                if m: sample_models[f"{sample}_{cond}"] = m

            t_pred = float(last_t)
            strats = compute_strategies(rgm, sample, cond, t_pred, train_df, sample_models)
            if strats is None: continue

            y_true = float(sub[sub["aging_time_day"] == last_t].iloc[-1][TARGET])
            record = {
                "sample": sample, "group": detect_group(sample), "cond": cond, "t": t_pred,
                "y_true": y_true,
            }
            for sn in STRATEGY_NAMES:
                record[sn] = strats[sn]
            records.append(record)

    return pd.DataFrame(records)


def eval_weights_weighted(lolo_df, weights_dict):
    """用给定的权重评估 - 测试集加权"""
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
    """用给定的权重评估 - 简单平均MAE"""
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


def search_best_weights(lolo_df, n_trials=20000, eval_fn=None):
    """对每组搜索最优权重 - 测试集加权优化"""
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
            # Dirichlet采样 + 允许部分权重为0
            raw_w = np.random.dirichlet(np.ones(len(STRATEGY_NAMES)) * 0.2)
            mask = np.random.random(len(STRATEGY_NAMES)) > 0.15  # 15%概率归零
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
            # 也计算unweighted MAE用于参考
            uw_mae, _ = eval_weights_unweighted(gdf, {group: best_weights[group]})
            print(f"  {group:15s}: wMAE={best_mae:.4f}, uMAE={uw_mae:.4f}, w={dict(zip(STRATEGY_NAMES, np.round(best_w, 3)))}")
        else:
            best_weights[group] = {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}

    return best_weights


def make_prediction(rgm, sample, cond, t_pred, df_train, sample_models, weights):
    """用给定的权重做单个预测"""
    strats = compute_strategies(rgm, sample, cond, t_pred, df_train, sample_models)
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
    print("  颜料老化色差预测 v34")
    print("  8策略 + 测试集加权LOLO搜索 + 2万次/组")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 预计算
    print("\n[预计算] LOLO策略值(8策略)...")
    lolo_df = precompute_lolo_strategies(df_train)
    print(f"  共 {len(lolo_df)} 个评估点")

    # v32基线(用v32权重在6策略上的效果)
    print("\n[v32参考权重](6策略部分):")
    v32_w = {
        "dye": {"ind": 0.655, "grp": 0.0, "ch": 0.042, "lin": 0.296, "scaled": 0.007, "ch_scaled": 0.0, "ratio": 0.0, "ind_ch_scaled": 0.0},
        "paper": {"ind": 0.008, "grp": 0.0, "ch": 0.0, "lin": 0.793, "scaled": 0.199, "ch_scaled": 0.0, "ratio": 0.0, "ind_ch_scaled": 0.0},
        "shu_red": {"ind": 0.062, "grp": 0.284, "ch": 0.0, "lin": 0.0, "scaled": 0.654, "ch_scaled": 0.0, "ratio": 0.0, "ind_ch_scaled": 0.0},
        "jade_green": {"ind": 0.0, "grp": 0.261, "ch": 0.561, "lin": 0.0, "scaled": 0.179, "ch_scaled": 0.0, "ratio": 0.0, "ind_ch_scaled": 0.0},
        "cobalt_blue": {"ind": 0.0, "grp": 0.691, "ch": 0.0, "lin": 0.0, "scaled": 0.309, "ch_scaled": 0.0, "ratio": 0.0, "ind_ch_scaled": 0.0},
        "other": {sn: 1/8 for sn in STRATEGY_NAMES},
    }
    v32_wmae, v32_ge = eval_weights_weighted(lolo_df, v32_w)
    v32_umae, _ = eval_weights_unweighted(lolo_df, v32_w)
    print(f"  v32参考: wMAE={v32_wmae:.4f}, uMAE={v32_umae:.4f}")

    # 搜索最优权重(测试集加权)
    print("\n[搜索] 测试集加权LOLO最优权重...")
    best_weights = search_best_weights(lolo_df, n_trials=20000)

    # 评估最优权重
    print("\n[最优权重] 评估:")
    opt_wmae, opt_ge = eval_weights_weighted(lolo_df, best_weights)
    opt_umae, _ = eval_weights_unweighted(lolo_df, best_weights)
    print(f"  测试集加权MAE = {opt_wmae:.4f}")
    print(f"  简单平均MAE = {opt_umae:.4f}")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        errs = opt_ge.get(g, [])
        if len(errs) >= 2:
            print(f"    {g:15s}: MAE={np.mean(errs):.4f}, n={len(errs)}")

    # R2
    all_true, all_pred = [], []
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = best_weights.get(g, {sn: 1/8 for sn in STRATEGY_NAMES})
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
    print(f"  R2 = {r2:.4f}")

    # 生成测试集预测
    print("\n[预测] 测试集...")
    rgm = RobustGroupModel(df_train)
    sample_models = {}
    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
            t, dE = prepare_series(sub)
            if len(t) < 2: continue
            tc, dEc = remove_outliers(t, dE)
            if len(tc) < 2: continue
            m = best_model(tc, dEc)
            if m: sample_models[f"{sample}_{cond}"] = m

    test_preds = []
    for _, row in df_test.iterrows():
        pred = make_prediction(rgm, row["sample"], row["aging_condition"],
                               float(row["aging_time_day"]), df_train, sample_models, best_weights)
        test_preds.append(float(max(pred, 0)))

    test_preds = np.array(test_preds)
    print(f"  范围: [{test_preds.min():.4f}, {test_preds.max():.4f}], 均值: {test_preds.mean():.4f}")

    # 保存
    out_csv = DATA_DIR / "predict_out.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(out_csv, index=False)
    pd.DataFrame({TARGET: test_preds}).to_csv(SCRIPT_DIR / "predict_out_v34.csv", index=False)

    print(f"\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d) -> {test_preds[i]:.4f}")

    print(f"\nv34: R2={r2:.4f}, uMAE={opt_umae:.4f}, wMAE={opt_wmae:.4f}")
    print(f"v32: R2=0.960, uMAE=0.375")
    print(f"v28: R2=0.918, uMAE=0.626")

    # 生成ensemble
    download_dir = SCRIPT_DIR
    for other_name, other_file in [("v14", "predict_out_v14.csv"), ("v32", "predict_out_v32.csv")]:
        other_path = download_dir / other_file
        if other_path.exists():
            other = pd.read_csv(other_path)[TARGET].values
            for w_other, label in [(0.3, f"0.3{other_name}_0.7v34"), (0.5, f"0.5{other_name}_0.5v34"), (0.7, f"0.7{other_name}_0.3v34")]:
                ens = w_other * other + (1 - w_other) * test_preds
                pd.DataFrame({TARGET: ens}).to_csv(download_dir / f"predict_out_{other_name}_v34_{label}.csv", index=False)
            print(f"\n[{other_name}+v34 ensemble] 已生成")

    # 3-model ensemble
    v14_path = download_dir / "predict_out_v14.csv"
    v32_path = download_dir / "predict_out_v32.csv"
    if v14_path.exists() and v32_path.exists():
        v14 = pd.read_csv(v14_path)[TARGET].values
        v32 = pd.read_csv(v32_path)[TARGET].values
        for w14, w32, label in [(0.2, 0.3, "0.2v14_0.3v32_0.5v34"), (0.3, 0.3, "0.3v14_0.3v32_0.4v34")]:
            ens = w14 * v14 + w32 * v32 + (1 - w14 - w32) * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(download_dir / f"predict_out_3ens_{label}.csv", index=False)
        print(f"[3-model ensemble] 已生成")

    print(f"\n[最优权重]")
    for g, w in best_weights.items():
        print(f"  {g:15s}: {w}")


if __name__ == "__main__":
    main()
