"""
颜料老化色差预测 - v37
基于v36的改进版：修复paper过估计 + 曙红约束建模 + scipy权重优化

核心改进（vs v36）:
1. 修复paper通道线性外推过估计（长程外推用sqrt而非线性）
2. 曙红组增加约束幂律策略（固定n∈{0.3,0.4,0.5,0.6,0.7}）
3. 非单调数据检测（dE下降时用组级轨迹替代个体模型）
4. scipy.optimize L-BFGS-B权重搜索（替代随机Dirichlet）
5. 策略可用性掩码（<3点不用饱和模型）
6. 加权目标函数（wMAE + λ*penalty防止过拟合）
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit, minimize
from sklearn.metrics import r2_score

# ===================== 路径配置 =====================
SCRIPT_DIR = Path("/home/z/my-project/download")
DATA_DIR = Path("/home/z/my-project/upload/baseline_and_data")
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
TARGET = "dietaE"

TEST_WEIGHTS = {
    "dye": 5, "paper": 4, "shu_red": 14, "jade_green": 7, "cobalt_blue": 7
}
TOTAL_TEST = sum(TEST_WEIGHTS.values())
TEST_GROUPS = list(TEST_WEIGHTS.keys())


# ===================== 基础工具函数 =====================
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


def is_monotonic_increasing(t, y):
    """检查dE是否单调递增"""
    if len(y) < 2: return True
    # 允许小幅下降（<10%）
    for i in range(1, len(y)):
        if y[i] < y[i-1] * 0.9:
            return False
    return True


def physical_max_dE(group):
    bounds = {
        "dye": 50, "paper": 25, "shu_red": 15,
        "jade_green": 15, "cobalt_blue": 15, "other": 30,
    }
    return bounds.get(group, 50)


# ===================== 扩展模型库 =====================
def fit_power_law(t, y, n_min=0.1, n_max=1.5):
    """dE = A * t^n"""
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
    res = minimize_scalar(neg_r2, bounds=(n_min, n_max), method="bounded")
    best_n = res.x
    tn = np.power(t, best_n)
    A = np.dot(tn, y) / (np.dot(tn, tn) + 1e-9)
    return {"type": "power", "A": A, "n": best_n, "score": -res.fun}


def fit_constrained_power(t, y, fixed_n):
    """dE = A * t^fixed_n, 只拟合A"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 2: return None
    t, y = t[mask], y[mask]
    tn = np.power(t, fixed_n)
    A = np.dot(tn, y) / (np.dot(tn, tn) + 1e-9)
    pred = A * tn
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    score = 1 - ss_res / (ss_tot + 1e-9)
    if score < -0.5: return None
    return {"type": "power", "A": A, "n": fixed_n, "score": score}


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
        return {"type": "mm", "A": popt[0] * y_max, "B": popt[1] * t_max, "score": score}
    except: return None


def fit_sqrt(t, y):
    mask = (t > 0)
    if mask.sum() < 2: return None
    t, y = t[mask], y[mask]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            np.vstack([np.sqrt(t), np.ones_like(t)]).T, y, rcond=None
        )
        pred = coeffs[0] * np.sqrt(t) + coeffs[1]
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
        popt, _ = curve_fit(
            expdec, t / t.max(), y / y_max, p0=[1.0], bounds=(0.01, 10), maxfev=5000
        )
        k = popt[0]
        pred = y_max * expdec(t / t.max(), k)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "exp_decay", "A": y_max, "k": k / t.max(), "score": score}
    except: return None


def fit_logistic(t, y):
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3: return None
    t, y = t[mask], y[mask]
    try:
        t_max = t.max()
        y_max = y.max() * 1.3 + 1e-6
        def logistic_norm(tn, k, t0):
            return 1.0 / (1.0 + np.exp(-k * (tn - t0)))
        popt, _ = curve_fit(
            logistic_norm, t / t_max, y / y_max,
            p0=[2.0, 0.5], bounds=([0.1, -1.0], [15.0, 3.0]), maxfev=10000
        )
        k, t0 = popt
        t0_real = t0 * t_max
        L = y_max
        def logistic_full(tn, A2):
            return A2 / (1.0 + np.exp(-k * (tn - t0_real)))
        popt2, _ = curve_fit(logistic_full, t, y, p0=[L], maxfev=5000)
        L = popt2[0]
        pred = logistic_full(t, L)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "logistic", "L": L, "k": k / t_max, "t0": t0_real, "score": score}
    except: return None


def fit_stretched_exp(t, y):
    mask = (t > 0) & (y >= 0)
    if mask.sum() < 4: return None
    t, y = t[mask], y[mask]
    try:
        t_max = t.max()
        y_max = y.max() + 1e-6
        def stretched_norm(tn, k, n):
            return 1.0 - np.exp(-np.power(k * tn + 1e-9, n))
        popt, _ = curve_fit(
            stretched_norm, t / t_max, y / y_max,
            p0=[1.0, 0.5], bounds=([0.01, 0.1], [10.0, 2.0]), maxfev=10000
        )
        k, n = popt
        k_real = k / t_max
        def stretched_full(tn, A2):
            return A2 * (1.0 - np.exp(-np.power(k_real * tn + 1e-9, n)))
        popt2, _ = curve_fit(stretched_full, t, y, p0=[y_max], maxfev=5000)
        A = popt2[0]
        pred = stretched_full(t, A)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "stretched_exp", "A": A, "k": k_real, "n": n, "score": score}
    except: return None


def predict_model(model, t_pred):
    if model is None: return None
    tp = model["type"]
    if tp == "power": return max(model["A"] * (t_pred ** model["n"]), 0)
    elif tp == "linear": return max(model["a"] + model["b"] * t_pred, 0)
    elif tp == "log": return model["A"] * np.log(1 + model["k"] * t_pred)
    elif tp == "mm": return max(model["A"] * t_pred / (model["B"] + t_pred), 0)
    elif tp == "sqrt": return max(model["A"] * np.sqrt(t_pred) + model["B"], 0)
    elif tp == "exp_decay": return max(model["A"] * (1 - np.exp(-model["k"] * t_pred)), 0)
    elif tp == "logistic":
        return model["L"] / (1.0 + np.exp(-model["k"] * (t_pred - model["t0"])))
    elif tp == "stretched_exp":
        return model["A"] * (1.0 - np.exp(-np.power(model["k"] * t_pred + 1e-9, model["n"])))
    return None


def best_model(t, y, min_score=-1.0):
    models = [
        fit_power_law(t, y), fit_log(t, y), fit_linear(t, y),
        fit_mm(t, y), fit_sqrt(t, y), fit_exp_decay(t, y),
        fit_logistic(t, y), fit_stretched_exp(t, y),
    ]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models: return None
    return max(models, key=lambda m: m["score"])


def best_saturation_model(t, y, min_score=-1.0):
    models = [fit_mm(t, y), fit_exp_decay(t, y), fit_logistic(t, y), fit_stretched_exp(t, y)]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models: return None
    return max(models, key=lambda m: m["score"])


def constrained_power_ensemble(t, y, n_values=None):
    """多个固定n的power law集成"""
    if n_values is None:
        n_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    models = [fit_constrained_power(t, y, n) for n in n_values]
    models = [m for m in models if m is not None]
    if not models: return None
    # 按R²加权平均
    total_w = sum(max(m["score"], 0.01) for m in models)
    if total_w <= 0: return None
    return {"type": "constrained_ensemble", "models": models, "total_weight": total_w}


def predict_constrained_ensemble(model, t_pred):
    if model is None or model["type"] != "constrained_ensemble": return None
    pred = sum(max(m["score"], 0.01) * predict_model(m, t_pred) for m in model["models"])
    return max(pred / model["total_weight"], 0)


# ===================== 稳健组级模型（修复通道外推） =====================
class RobustGroupModel:
    def __init__(self, df_train):
        self.group_models = {}
        self.group_models_all = {}
        self.group_sat_models = {}
        self.group_medians = {}
        self.group_channel_medians = {}
        self.group_channel_models = {}
        self.group_ratios = {}
        self.group_constrained = {}  # 组级约束幂律集成
        self._build(df_train)

    def _build(self, df_train):
        all_groups = ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]

        for group in all_groups:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]

            # 组级增长比例
            group_ratios = []
            for sample in members:
                for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
                    sub = df_train[
                        (df_train["sample"] == sample) &
                        (df_train["aging_condition"] == cond)
                    ].sort_values("aging_time_day")
                    if len(sub) >= 2:
                        times = sub["aging_time_day"].values
                        dEs = sub[TARGET].values
                        for i in range(1, len(times)):
                            if dEs[i - 1] > 0.01:
                                group_ratios.append(dEs[i] / dEs[i - 1])
            if group_ratios:
                clean = [r for r in group_ratios if 0.5 < r < 5.0]
                self.group_ratios[group] = clean

            for cond in ["UV", "humid-_heat"]:
                time_data, channel_data = {}, {}
                for m in members:
                    sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == cond)]
                    sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
                    for _, row in sub.iterrows():
                        t = int(row["aging_time_day"])
                        if t not in time_data:
                            time_data[t] = []
                        time_data[t].append(row[TARGET])
                        if t not in channel_data:
                            channel_data[t] = {"dL": [], "da": [], "db": []}
                        channel_data[t]["dL"].append(row["L"] - row["L0"])
                        channel_data[t]["da"].append(row["a"] - row["a0"])
                        channel_data[t]["db"].append(row["b"] - row["b0"])

                times = sorted(time_data.keys())
                if len(times) < 2: continue
                medians_t = np.array(times, dtype=float)
                medians_y = np.array([np.median(time_data[t]) for t in times])
                key = f"{group}_{cond}"
                self.group_medians[key] = {"times": medians_t, "medians": medians_y}

                mt, my = remove_outliers(medians_t, medians_y)
                if len(mt) >= 2:
                    model = best_model(mt, my)
                    if model:
                        self.group_models[key] = model

                    all_ms = [
                        fit_power_law(mt, my), fit_log(mt, my), fit_linear(mt, my),
                        fit_mm(mt, my), fit_sqrt(mt, my), fit_exp_decay(mt, my) if len(mt) >= 3 else None,
                        fit_logistic(mt, my) if len(mt) >= 3 else None,
                        fit_stretched_exp(mt, my) if len(mt) >= 4 else None,
                    ]
                    all_ms = [m for m in all_ms if m is not None and m.get("score", -999) > 0]
                    all_ms.sort(key=lambda m: -m["score"])
                    if all_ms:
                        self.group_models_all[key] = all_ms[:4]

                    sat = best_saturation_model(mt, my, min_score=-2.0)
                    if sat:
                        self.group_sat_models[key] = sat

                    # 组级约束幂律集成
                    cpe = constrained_power_ensemble(mt, my)
                    if cpe:
                        self.group_constrained[key] = cpe

                # 通道级建模
                ch_times = sorted(channel_data.keys())
                ch_meds = {
                    t: {
                        "dL": np.median(channel_data[t]["dL"]),
                        "da": np.median(channel_data[t]["da"]),
                        "db": np.median(channel_data[t]["db"]),
                    }
                    for t in ch_times
                }
                self.group_channel_medians[key] = ch_times, ch_meds

                ch_models_dict = {}
                for ch_name in ["dL", "da", "db"]:
                    ch_vals = np.array([ch_meds[t][ch_name] for t in ch_times])
                    lin = fit_linear(medians_t, ch_vals)
                    # 新增：sqrt模型用于长程外推（比线性保守）
                    sqrt_model = fit_sqrt(medians_t, ch_vals)
                    # 饱和模型
                    sat_ch = fit_mm(medians_t, np.abs(ch_vals) + 1e-9) if len(ch_times) >= 3 else None
                    abs_model = best_model(medians_t, np.abs(ch_vals) + 1e-9, min_score=-2.0)
                    ch_models_dict[ch_name] = {
                        "linear": lin,
                        "sqrt": sqrt_model,
                        "abs_model": abs_model,
                        "sat_ch": sat_ch,
                        "sign": np.sign(np.median(ch_vals)),
                    }
                self.group_channel_models[key] = ch_models_dict

    def predict_group_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key in self.group_models:
            return predict_model(self.group_models[key], t_pred)
        return None

    def predict_group_constrained(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key in self.group_constrained:
            return predict_constrained_ensemble(self.group_constrained[key], t_pred)
        return None

    def predict_group_dE_saturation(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key in self.group_sat_models:
            return predict_model(self.group_sat_models[key], t_pred)
        return None

    def predict_group_channel_dE(self, group, cond, t_pred):
        """改进的通道分解预测：长程外推用sqrt/饱和模型替代线性"""
        key = f"{group}_{cond}"
        if key not in self.group_channel_medians:
            return None
        times, ch_meds = self.group_channel_medians[key]
        if not times:
            return None

        t_max_train = max(times)
        extrap_factor = t_pred / t_max_train if t_max_train > 0 else 1.0

        # 外推
        if key in self.group_channel_models and t_pred > times[-1]:
            ch_preds = {}
            for ch_name in ["dL", "da", "db"]:
                ch_model = self.group_channel_models[key][ch_name]

                if extrap_factor > 2.0:
                    # 长程外推(>2x): 用sqrt或饱和模型，不用线性
                    p_sqrt = predict_model(ch_model["sqrt"], t_pred) if ch_model["sqrt"] else None
                    p_sat = None
                    if ch_model["sat_ch"]:
                        p_sat_val = predict_model(ch_model["sat_ch"], t_pred)
                        if p_sat_val is not None:
                            p_sat = ch_model["sign"] * p_sat_val
                    p_abs = None
                    if ch_model["abs_model"]:
                        p_abs_val = predict_model(ch_model["abs_model"], t_pred)
                        if p_abs_val is not None:
                            p_abs = ch_model["sign"] * p_abs_val

                    # 优先sqrt > 饱和 > 线性
                    if p_sqrt is not None:
                        ch_preds[ch_name] = p_sqrt
                    elif p_sat is not None:
                        ch_preds[ch_name] = p_sat
                    elif p_abs is not None:
                        ch_preds[ch_name] = p_abs
                    else:
                        ch_preds[ch_name] = ch_meds[times[-1]][ch_name]
                else:
                    # 短程外推: 用线性
                    p_lin = predict_model(ch_model["linear"], t_pred)
                    if p_lin is not None:
                        ch_preds[ch_name] = p_lin
                    else:
                        ch_preds[ch_name] = ch_meds[times[-1]][ch_name]

            if len(ch_preds) == 3:
                dE = np.sqrt(ch_preds["dL"] ** 2 + ch_preds["da"] ** 2 + ch_preds["db"] ** 2)
                return float(max(dE, 0))

        # 插值
        if t_pred <= times[0]:
            ch = ch_meds[times[0]]
        elif t_pred >= times[-1]:
            ch = ch_meds[times[-1]]
        else:
            ch = ch_meds[times[-1]]
            for i in range(len(times) - 1):
                if times[i] <= t_pred <= times[i + 1]:
                    frac = (t_pred - times[i]) / (times[i + 1] - times[i])
                    ch = {
                        k: ch_meds[times[i]][k] * (1 - frac) + ch_meds[times[i + 1]][k] * frac
                        for k in ["dL", "da", "db"]
                    }
                    break
        return float(max(np.sqrt(ch["dL"] ** 2 + ch["da"] ** 2 + ch["db"] ** 2), 0))

    def get_group_ratio(self, group):
        ratios = self.group_ratios.get(group, [])
        if ratios:
            return np.median(ratios)
        return 1.2


# ===================== 11种预测策略 =====================
STRATEGY_NAMES = [
    "ind", "grp", "ch", "lin", "scaled", "ratio",
    "ind_ch_scaled", "cross_sample", "saturation", "conservative",
    "constrained_pow"  # 新增：约束幂律集成
]


def compute_strategies(rgm, sample, cond, t_pred, df_train, sample_models=None):
    sub = df_train[
        (df_train["sample"] == sample) & (df_train["aging_condition"] == cond)
    ].sort_values("aging_time_day")
    if len(sub) == 0: return None
    t_arr, dE_arr = prepare_series(sub)
    if len(t_arr) == 0: return None
    tc, dEc = remove_outliers(t_arr, dE_arr)
    group = detect_group(sample)
    key = f"{sample}_{cond}"
    monotonic = is_monotonic_increasing(tc, dEc)

    # ---- 策略1: ind ----
    p_ind = None
    if sample_models and key in sample_models:
        # 非单调数据时降低个体策略权重（返回None让其他策略接管）
        if monotonic or len(tc) >= 3:
            p_ind = predict_model(sample_models[key], t_pred)

    # ---- 策略2: grp ----
    p_grp = rgm.predict_group_dE(group, cond, t_pred)

    # ---- 策略3: ch（修复版） ----
    p_ch = rgm.predict_group_channel_dE(group, cond, t_pred)

    # ---- 策略4: lin ----
    p_lin = None
    if len(tc) >= 2 and monotonic:
        rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
        p_lin = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)

    # ---- 策略5: scaled ----
    p_scaled = None
    if p_grp is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_medians:
            gm = rgm.group_medians[gmk]
            gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
            if gv > 0.01:
                p_scaled = p_grp * dEc[-1] / gv

    # ---- 策略6: ratio ----
    p_ratio = None
    if len(tc) >= 1 and t_pred > tc[-1] and monotonic:
        ratio_per_step = rgm.get_group_ratio(group)
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_medians:
            gm = rgm.group_medians[gmk]
            if len(gm["times"]) >= 2:
                steps = np.diff(gm["times"])
                avg_step = np.mean(steps)
                if avg_step > 0:
                    n_steps = (t_pred - tc[-1]) / avg_step
                    p_ratio = dEc[-1] * (ratio_per_step ** n_steps)

    # ---- 策略7: ind_ch_scaled ----
    p_ind_ch_scaled = None
    if p_ch is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_channel_medians:
            ch_times, ch_meds = rgm.group_channel_medians[gmk]
            if len(ch_times) >= 1:
                t_ch_sub = df_train[
                    (df_train["sample"] == sample) & (df_train["aging_condition"] == cond)
                ]
                t_ch_sub = t_ch_sub[t_ch_sub["aging_time_day"] > 0].sort_values("aging_time_day")
                if len(t_ch_sub) > 0:
                    last_row = t_ch_sub.iloc[-1]
                    ind_dL = last_row["L"] - last_row["L0"]
                    ind_da = last_row["a"] - last_row["a0"]
                    ind_db = last_row["b"] - last_row["b0"]
                    ind_dE = np.sqrt(ind_dL ** 2 + ind_da ** 2 + ind_db ** 2)

                    closest_t = ch_times[np.argmin(np.abs(np.array(ch_times) - tc[-1]))]
                    grp_ch = ch_meds[closest_t]
                    grp_dE = np.sqrt(grp_ch["dL"] ** 2 + grp_ch["da"] ** 2 + grp_ch["db"] ** 2)

                    if grp_dE > 0.01:
                        scale = ind_dE / grp_dE
                        p_ind_ch_scaled = p_ch * scale

    # ---- 策略8: cross_sample ----
    p_cross = None
    members = [
        s for s in df_train["sample"].unique()
        if detect_group(s) == group and s != sample
    ]
    if members:
        cross_preds = []
        for other_sample in members:
            other_sub = df_train[
                (df_train["sample"] == other_sample) &
                (df_train["aging_condition"] == cond)
            ]
            other_t, other_dE = prepare_series(other_sub)
            if len(other_t) < 2:
                continue
            other_tc, other_dEc = remove_outliers(other_t, other_dE)
            if len(other_tc) < 2:
                continue
            other_model = best_model(other_tc, other_dEc)
            if other_model:
                p = predict_model(other_model, t_pred)
                if p is not None and np.isfinite(p) and p > 0:
                    cross_preds.append(p)
        if cross_preds:
            p_cross = np.median(cross_preds)

    # ---- 策略9: saturation ----
    p_sat = rgm.predict_group_dE_saturation(group, cond, t_pred)
    if len(tc) >= 3:
        ind_sat = best_saturation_model(tc, dEc, min_score=-2.0)
        if ind_sat:
            p_ind_sat = predict_model(ind_sat, t_pred)
            if p_ind_sat is not None and np.isfinite(p_ind_sat):
                if p_sat is not None:
                    p_sat = (p_sat + p_ind_sat) / 2.0
                else:
                    p_sat = p_ind_sat

    # ---- 策略10: conservative ----
    p_conservative = None
    all_valid = []
    for p in [p_ind, p_grp, p_ch, p_scaled, p_sat, p_cross]:
        if p is not None and np.isfinite(p) and p > 0:
            all_valid.append(p)
    if all_valid:
        p_conservative = np.exp(np.mean(np.log(all_valid)))

    # ---- 策略11: constrained_pow（新增） ----
    p_constrained = rgm.predict_group_constrained(group, cond, t_pred)
    # 也尝试个体约束幂律
    if len(tc) >= 2:
        ind_cpe = constrained_power_ensemble(tc, dEc)
        if ind_cpe:
            p_ind_cpe = predict_constrained_ensemble(ind_cpe, t_pred)
            if p_ind_cpe is not None and np.isfinite(p_ind_cpe):
                if p_constrained is not None:
                    p_constrained = (p_constrained + p_ind_cpe) / 2.0
                else:
                    p_constrained = p_ind_cpe

    # 应用物理上界
    pmax = physical_max_dE(group)
    def clamp(val):
        if val is None: return None
        return min(max(val, 0), pmax)

    return {
        "ind": clamp(p_ind),
        "grp": clamp(p_grp),
        "ch": clamp(p_ch),
        "lin": clamp(p_lin),
        "scaled": clamp(p_scaled),
        "ratio": clamp(p_ratio),
        "ind_ch_scaled": clamp(p_ind_ch_scaled),
        "cross_sample": clamp(p_cross),
        "saturation": clamp(p_sat),
        "conservative": clamp(p_conservative),
        "constrained_pow": clamp(p_constrained),
    }


# ===================== LOLO 预计算 =====================
def precompute_lolo_strategies(df_train):
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
            t_sub = train_df[
                (train_df["sample"] == sample) & (train_df["aging_condition"] == cond)
            ]
            t_arr, dE_arr = prepare_series(t_sub)
            if len(t_arr) == 0: continue
            tc, dEc = remove_outliers(t_arr, dE_arr)

            sample_models = {}
            if len(tc) >= 2:
                m = best_model(tc, dEc)
                if m:
                    sample_models[f"{sample}_{cond}"] = m

            t_pred = float(last_t)
            strats = compute_strategies(rgm, sample, cond, t_pred, train_df, sample_models)
            if strats is None: continue

            y_true = float(sub[sub["aging_time_day"] == last_t].iloc[-1][TARGET])
            record = {
                "sample": sample, "group": detect_group(sample),
                "cond": cond, "t": t_pred, "y_true": y_true,
            }
            for sn in STRATEGY_NAMES:
                record[sn] = strats[sn]
            records.append(record)

    return pd.DataFrame(records)


# ===================== scipy.optimize 权重搜索 =====================
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
            if g not in group_errors:
                group_errors[g] = []
            group_errors[g].append(err)
    if not weighted_errors: return float("inf"), group_errors
    return sum(weighted_errors) / TOTAL_TEST, group_errors


def eval_weights_unweighted(lolo_df, weights_dict):
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
            if g not in group_errors:
                group_errors[g] = []
            group_errors[g].append(err)
    return np.mean(all_errors) if all_errors else float("inf"), group_errors


def softmax_to_weights(x):
    """softmax参数化：任意实数x → 有效权重"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def search_best_weights_scipy(lolo_df):
    """用scipy.optimize L-BFGS-B搜索最优权重"""
    n_strats = len(STRATEGY_NAMES)
    best_weights = {}

    for group in TEST_GROUPS:
        gdf = lolo_df[lolo_df["group"] == group]
        if len(gdf) < 2:
            best_weights[group] = {sn: 1/n_strats for sn in STRATEGY_NAMES}
            continue

        def objective(x):
            w_dict = dict(zip(STRATEGY_NAMES, softmax_to_weights(x)))
            mae, _ = eval_weights_weighted(gdf, {group: w_dict})
            # 加L2正则防止极端权重
            w = softmax_to_weights(x)
            reg = 0.001 * np.sum(w ** 2)
            return mae + reg

        best_mae = float("inf")
        best_w = None

        # 多起点优化
        for trial in range(20):
            if trial == 0:
                x0 = np.zeros(n_strats)
            else:
                x0 = np.random.randn(n_strats) * 1.5

            res = minimize(objective, x0, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-10})

            w_dict = dict(zip(STRATEGY_NAMES, softmax_to_weights(res.x)))
            mae, _ = eval_weights_weighted(gdf, {group: w_dict})

            if mae < best_mae:
                best_mae = mae
                best_w = softmax_to_weights(res.x).copy()

        # 用差分进化精调
        def de_objective(x):
            w_dict = dict(zip(STRATEGY_NAMES, softmax_to_weights(x)))
            mae, _ = eval_weights_weighted(gdf, {group: w_dict})
            w = softmax_to_weights(x)
            reg = 0.001 * np.sum(w ** 2)
            return mae + reg

        bounds = [(-3, 3)] * n_strats
        de_res = minimize(de_objective, best_w, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})

        w_final = softmax_to_weights(de_res.x)
        w_dict = dict(zip(STRATEGY_NAMES, w_final))
        mae_final, _ = eval_weights_weighted(gdf, {group: w_dict})

        best_weights[group] = w_dict
        uw_mae, _ = eval_weights_unweighted(gdf, {group: w_dict})
        top_strats = sorted(zip(STRATEGY_NAMES, w_final), key=lambda x: -x[1])[:4]
        top_str = {k: round(v, 3) for k, v in top_strats if v > 0.01}
        print(f"  {group:15s}: wMAE={mae_final:.4f}, uMAE={uw_mae:.4f}, top={top_str}")

    best_weights["other"] = {sn: 1/n_strats for sn in STRATEGY_NAMES}
    return best_weights


# ===================== 预测和评估 =====================
def make_prediction(rgm, sample, cond, t_pred, df_train, sample_models, weights):
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


def compute_r2_and_pred(lolo_df, weights):
    all_true, all_pred = [], []
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = weights.get(g, {sn: 1/len(STRATEGY_NAMES) for sn in STRATEGY_NAMES})
        ws, wp = 0, 0
        for sn in STRATEGY_NAMES:
            v = row[sn]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ws += w.get(sn, 0)
                wp += w.get(sn, 0) * v
        if ws > 0:
            all_true.append(row["y_true"])
            all_pred.append(wp / ws)
    return np.array(all_true), np.array(all_pred)


# ===================== 主函数 =====================
def main():
    print("=" * 70)
    print("  颜料老化色差预测 v37")
    print("  修复paper过估计 + 曙红约束建模 + scipy权重优化 + 非单调检测")
    print("=" * 70)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    print("\n[训练数据概览]")
    for group in TEST_GROUPS:
        sub = df_train[df_train["sample"].apply(detect_group) == group]
        samples = sub["sample"].unique()
        times = sorted(sub[sub["aging_time_day"] > 0]["aging_time_day"].unique())
        print(f"  {group:15s}: {len(samples)} 样本, 时间={times}")
    other = df_train[df_train["sample"].apply(detect_group) == "other"]
    print(f"  {'other':15s}: {len(other['sample'].unique())} 样本 (不在测试集)")

    print("\n[步骤1] LOLO策略预计算 (11策略)...")
    lolo_df = precompute_lolo_strategies(df_train)
    print(f"  共 {len(lolo_df)} 个评估点")
    for g in TEST_GROUPS + ["other"]:
        n = len(lolo_df[lolo_df["group"] == g])
        if n > 0:
            print(f"    {g:15s}: {n} 个LOLO点")

    print(f"\n[步骤2] scipy.optimize权重搜索 (L-BFGS-B + Nelder-Mead)...")
    best_weights = search_best_weights_scipy(lolo_df)

    print("\n[步骤3] LOLO评估结果:")
    opt_wmae, opt_ge = eval_weights_weighted(lolo_df, best_weights)
    opt_umae, _ = eval_weights_unweighted(lolo_df, best_weights)

    test_lolo_df = lolo_df[lolo_df['group'].isin(TEST_GROUPS)]
    test_true, test_pred = compute_r2_and_pred(test_lolo_df, best_weights)
    r2 = r2_score(test_true, test_pred)
    test_umae = np.mean(np.abs(test_true - test_pred))

    all_true, all_pred = compute_r2_and_pred(lolo_df, best_weights)
    r2_all = r2_score(all_true, all_pred)

    print(f"\n  测试集组评估:")
    print(f"    测试集加权MAE = {opt_wmae:.4f}")
    print(f"    测试集组MAE   = {test_umae:.4f}")
    print(f"    测试集组R²    = {r2:.4f}")
    print(f"  全部组评估:")
    print(f"    简单平均MAE   = {opt_umae:.4f}")
    print(f"    全部R²        = {r2_all:.4f}")

    print(f"\n  各组MAE:")
    for g in TEST_GROUPS + ["other"]:
        errs = opt_ge.get(g, [])
        if len(errs) >= 1:
            print(f"    {g:15s}: MAE={np.mean(errs):.4f} (n={len(errs)})")

    print("\n[步骤4] 测试集预测...")
    rgm = RobustGroupModel(df_train)

    sample_models = {}
    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[
                (df_train["sample"] == sample) & (df_train["aging_condition"] == cond)
            ]
            t, dE = prepare_series(sub)
            if len(t) < 2: continue
            tc, dEc = remove_outliers(t, dE)
            if len(tc) < 2: continue
            m = best_model(tc, dEc)
            if m:
                sample_models[f"{sample}_{cond}"] = m

    test_preds = []
    for _, row in df_test.iterrows():
        pred = make_prediction(
            rgm, row["sample"], row["aging_condition"],
            float(row["aging_time_day"]), df_train, sample_models, best_weights
        )
        test_preds.append(float(max(pred, 0)))

    test_preds = np.array(test_preds)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    print("\n[步骤5] 保存预测结果...")
    pd.DataFrame({TARGET: test_preds}).to_csv(DATA_DIR / "predict_out.csv", index=False)
    pd.DataFrame({TARGET: test_preds}).to_csv(SCRIPT_DIR / "predict_out_v37.csv", index=False)
    print(f"  已保存: {DATA_DIR / 'predict_out.csv'}")
    print(f"  已保存: {SCRIPT_DIR / 'predict_out_v37.csv'}")

    print(f"\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        g = detect_group(row["sample"])
        print(
            f"  {row['sample']:20s} ({row['aging_condition']:12s}, "
            f"t={row['aging_time_day']:3.0f}d) [{g:11s}] -> {test_preds[i]:.4f}"
        )

    # 生成ensemble
    print("\n[步骤6] 生成ensemble文件...")
    for v_name in ["v14", "v34", "v36"]:
        v_path = SCRIPT_DIR / f"predict_out_{v_name}.csv"
        if v_path.exists():
            v_vals = pd.read_csv(v_path)[TARGET].values
            print(f"  找到{v_name}预测 ({len(v_vals)} 行)")
            for w_new, label in [(0.3, f"0.3{v_name}_0.7v37"), (0.5, f"0.5{v_name}_0.5v37"), (0.7, f"0.7{v_name}_0.3v37")]:
                ens = w_new * v_vals + (1 - w_new) * test_preds
                pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_{label}.csv", index=False)

    # 4-model ensemble
    all_v = {}
    for v_name in ["v14", "v34", "v36"]:
        v_path = SCRIPT_DIR / f"predict_out_{v_name}.csv"
        if v_path.exists():
            all_v[v_name] = pd.read_csv(v_path)[TARGET].values
    if len(all_v) == 3:
        for label, ws in [("0.2v14_0.2v34_0.2v36_0.4v37", [0.2,0.2,0.2,0.4]),
                          ("0.15v14_0.15v34_0.15v36_0.55v37", [0.15,0.15,0.15,0.55]),
                          ("0.25v14_0.25v34_0.25v36_0.25v37", [0.25,0.25,0.25,0.25])]:
            ens = sum(w * all_v[k] for (k, w) in zip(["v14","v34","v36"], ws[:3])) + ws[3] * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_4ens_{label}.csv", index=False)
        print(f"  [4-model ensemble] 已生成文件")

    print("\n" + "=" * 70)
    print("  最终结果")
    print("=" * 70)
    print(f"\n  v37 LOLO评估:")
    print(f"    R2  = {r2:.4f}")
    print(f"    MAE = {opt_umae:.4f}")
    print(f"    wMAE= {opt_wmae:.4f}")
    print(f"\n  历史对比:")
    print(f"    v37: R2={r2:.4f}, MAE={opt_umae:.4f}")
    print(f"    v36: R2=0.9824, MAE=0.7352")
    print(f"    v34: R2=0.967, MAE=0.340")


if __name__ == "__main__":
    main()
