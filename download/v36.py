"""
颜料老化色差预测 - v36
基于v34的全面改进版：测试集感知 + 饱和外推 + 跨样本迁移

核心改进（vs v34）:
1. 移除"other"组优化 - 不在测试集中，节省计算
2. 更好的外推模型：logistic、stretched_exp等饱和模型，自然趋平
3. Bayesian先验约束：power law指数n限制在0.1-1.5（物理合理范围）
4. 测试集加权优化：只优化测试集涉及的5个组
5. 通道级物理建模：dL/da/db各有物理含义和有界行为
6. 10种策略：ind, grp, ch, lin, scaled, ratio, ind_ch_scaled, cross_sample, saturation, conservative
7. 50000次Dirichlet采样搜索最优权重
8. 物理上界约束：dE不应超过物理最大值
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== 路径配置 =====================
SCRIPT_DIR = Path("/home/z/my-project/download")
DATA_DIR = Path("/home/z/my-project/upload/baseline_and_data")
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
TARGET = "dietaE"

# 测试集各组样本数（不包含"other"组）
TEST_WEIGHTS = {
    "dye": 5, "paper": 4, "shu_red": 14, "jade_green": 7, "cobalt_blue": 7
}
TOTAL_TEST = sum(TEST_WEIGHTS.values())

# 只在测试集中出现的组
TEST_GROUPS = list(TEST_WEIGHTS.keys())


# ===================== 基础工具函数 =====================
def detect_group(sample: str) -> str:
    """根据样本名称检测所属组"""
    if "翡翠绿" in sample: return "jade_green"
    if "钴蓝" in sample: return "cobalt_blue"
    if "曙红" in sample: return "shu_red"
    if "皮纸" in sample: return "paper"
    if any(x in sample for x in ["染料", "紫草", "苏木", "红花", "黄檗"]):
        return "dye"
    return "other"


def prepare_series(df_sub):
    """提取时间序列数据（按时间聚合取均值）"""
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)


def prepare_channels(df_sub):
    """提取通道级时间序列数据"""
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
    """去除异常值（基于差分幅度）"""
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


def physical_max_dE(group):
    """物理上界：不同组的dE不应超过合理范围"""
    # CIELAB空间中dE的理论最大值约176，但实际老化远小于此
    # 根据组特征设定合理上界
    bounds = {
        "dye": 50,         # 染料老化较大变化
        "paper": 25,       # 纸张变化中等
        "shu_red": 15,     # 曙红变化较小
        "jade_green": 15,  # 翡翠绿变化较小
        "cobalt_blue": 15, # 钴蓝变化较小
        "other": 30,       # 其他颜料
    }
    return bounds.get(group, 50)


# ===================== 扩展模型库 =====================
# 8种生长模型 + 2种新增饱和模型

def fit_power_law(t, y):
    """dE = A * t^n, n约束在0.1-1.5（物理合理范围）"""
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

    res = minimize_scalar(neg_r2, bounds=(0.1, 1.5), method="bounded")
    best_n = res.x
    tn = np.power(t, best_n)
    A = np.dot(tn, y) / (np.dot(tn, tn) + 1e-9)
    return {"type": "power", "A": A, "n": best_n, "score": -res.fun}


def fit_linear(t, y):
    """dE = a + b*t"""
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
    """dE = A * log(1 + k*t)"""
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
    """Michaelis-Menten: dE = A*t / (B + t)  -- 饱和模型"""
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
    """dE = A*sqrt(t) + B"""
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
    """dE = A*(1 - exp(-k*t))  -- 饱和模型"""
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
    """Logistic: dE = L / (1 + exp(-k*(t-t0)))  -- S形饱和模型"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3: return None
    t, y = t[mask], y[mask]
    try:
        t_max = t.max()
        y_max = y.max() * 1.3 + 1e-6  # L > y_max to allow headroom

        def logistic_norm(tn, k, t0):
            return 1.0 / (1.0 + np.exp(-k * (tn - t0)))
        popt, _ = curve_fit(
            logistic_norm, t / t_max, y / y_max,
            p0=[2.0, 0.5], bounds=([0.1, -1.0], [15.0, 3.0]), maxfev=10000
        )
        k, t0 = popt
        t0_real = t0 * t_max
        L = y_max

        # 重新用原始尺度精确拟合L
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
    """拉伸指数: dE = A*(1 - exp(-(k*t)^n))  -- 灵活饱和模型"""
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
    """用模型预测"""
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
    """从所有模型中选择最佳"""
    models = [
        fit_power_law(t, y), fit_log(t, y), fit_linear(t, y),
        fit_mm(t, y), fit_sqrt(t, y), fit_exp_decay(t, y),
        fit_logistic(t, y), fit_stretched_exp(t, y),
    ]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models: return None
    return max(models, key=lambda m: m["score"])


def ensemble_predict(models, t_pred, min_score=-1.0):
    """多个模型按R2加权平均预测"""
    if not models:
        return None
    valid = []
    for m in models:
        if m is not None and m.get("score", -999) > min_score:
            p = predict_model(m, t_pred)
            if p is not None and np.isfinite(p):
                valid.append((m, p))
    if not valid:
        return None
    total_weight = sum(max(m["score"], 0.01) for m, _ in valid)
    if total_weight <= 0:
        return None
    pred = sum(max(m["score"], 0.01) * p for m, p in valid) / total_weight
    return max(pred, 0)


def best_saturation_model(t, y, min_score=-1.0):
    """只选择饱和模型（MM, exp_decay, logistic, stretched_exp）"""
    models = [fit_mm(t, y), fit_exp_decay(t, y), fit_logistic(t, y), fit_stretched_exp(t, y)]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models: return None
    return max(models, key=lambda m: m["score"])


# ===================== 稳健组级模型 =====================
class RobustGroupModel:
    """组级模型：支持中位数建模、通道建模、跨样本信息"""

    def __init__(self, df_train):
        self.group_models = {}           # group_cond -> best model
        self.group_models_all = {}       # group_cond -> list of top models
        self.group_sat_models = {}       # group_cond -> best saturation model
        self.group_medians = {}          # group_cond -> {times, medians}
        self.group_channel_medians = {}  # group_cond -> (times, ch_meds)
        self.group_channel_models = {}   # group_cond -> {ch_name: models}
        self.group_ratios = {}           # group -> list of growth ratios
        self._build(df_train)

    def _build(self, df_train):
        all_groups = ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]

        for group in all_groups:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]

            # 收集组级增长比例
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

                # 去除异常值
                mt, my = remove_outliers(medians_t, medians_y)
                if len(mt) >= 2:
                    # 最佳模型
                    model = best_model(mt, my)
                    if model:
                        self.group_models[key] = model

                    # 保存top-3模型用于ensemble
                    all_ms = [
                        fit_power_law(mt, my), fit_log(mt, my), fit_linear(mt, my),
                        fit_mm(mt, my), fit_sqrt(mt, my), fit_exp_decay(mt, my),
                        fit_logistic(mt, my), fit_stretched_exp(mt, my),
                    ]
                    all_ms = [m for m in all_ms if m is not None and m.get("score", -999) > 0]
                    all_ms.sort(key=lambda m: -m["score"])
                    if all_ms:
                        self.group_models_all[key] = all_ms[:4]

                    # 饱和模型
                    sat = best_saturation_model(mt, my, min_score=-2.0)
                    if sat:
                        self.group_sat_models[key] = sat

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
                    abs_model = best_model(medians_t, np.abs(ch_vals) + 1e-9, min_score=-2.0)
                    ch_models_dict[ch_name] = {
                        "linear": lin,
                        "abs_model": abs_model,
                        "sign": np.sign(np.median(ch_vals)),
                    }
                self.group_channel_models[key] = ch_models_dict

    def predict_group_dE(self, group, cond, t_pred):
        """组级dE最佳模型预测"""
        key = f"{group}_{cond}"
        if key in self.group_models:
            return predict_model(self.group_models[key], t_pred)
        return None

    def predict_group_dE_ensemble(self, group, cond, t_pred):
        """组级ensemble预测（top-4模型加权平均）"""
        key = f"{group}_{cond}"
        if key not in self.group_models_all:
            return None
        return ensemble_predict(self.group_models_all[key], t_pred)

    def predict_group_dE_saturation(self, group, cond, t_pred):
        """组级饱和模型预测（外推时更保守）"""
        key = f"{group}_{cond}"
        if key in self.group_sat_models:
            return predict_model(self.group_sat_models[key], t_pred)
        return None

    def predict_group_channel_dE(self, group, cond, t_pred):
        """组级通道分解预测"""
        key = f"{group}_{cond}"
        if key not in self.group_channel_medians:
            return None
        times, ch_meds = self.group_channel_medians[key]
        if not times:
            return None

        # 外推：用通道模型
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
                # 外推时优先用线性（通道变化通常较线性）
                if p_lin is not None:
                    ch_preds[ch_name] = p_lin
                elif p_abs is not None:
                    ch_preds[ch_name] = p_abs
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
            ch = ch_meds[times[-1]]  # fallback
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
        """获取组级增长比例中位数"""
        ratios = self.group_ratios.get(group, [])
        if ratios:
            return np.median(ratios)
        return 1.2


# ===================== 10种预测策略 =====================
STRATEGY_NAMES = [
    "ind", "grp", "ch", "lin", "scaled", "ratio",
    "ind_ch_scaled", "cross_sample", "saturation", "conservative"
]


def compute_strategies(rgm, sample, cond, t_pred, df_train, sample_models=None):
    """计算10种策略的预测值"""
    sub = df_train[
        (df_train["sample"] == sample) & (df_train["aging_condition"] == cond)
    ].sort_values("aging_time_day")
    if len(sub) == 0: return None
    t_arr, dE_arr = prepare_series(sub)
    if len(t_arr) == 0: return None
    tc, dEc = remove_outliers(t_arr, dE_arr)
    group = detect_group(sample)
    key = f"{sample}_{cond}"

    # ---- 策略1: ind - 个体最佳模型 ----
    p_ind = None
    if sample_models and key in sample_models:
        p_ind = predict_model(sample_models[key], t_pred)

    # ---- 策略2: grp - 组级最佳dE模型 ----
    p_grp = rgm.predict_group_dE(group, cond, t_pred)

    # ---- 策略3: ch - 组级通道分解 ----
    p_ch = rgm.predict_group_channel_dE(group, cond, t_pred)

    # ---- 策略4: lin - 线性外推(最后两点的增量) ----
    p_lin = None
    if len(tc) >= 2:
        rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
        p_lin = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)

    # ---- 策略5: scaled - 个体/组dE比例缩放组预测 ----
    p_scaled = None
    if p_grp is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_medians:
            gm = rgm.group_medians[gmk]
            gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
            if gv > 0.01:
                p_scaled = p_grp * dEc[-1] / gv

    # ---- 策略6: ratio - 增长比例外推 ----
    p_ratio = None
    if len(tc) >= 1 and t_pred > tc[-1]:
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

    # ---- 策略7: ind_ch_scaled - 个体通道缩放组通道预测 ----
    p_ind_ch_scaled = None
    if p_ch is not None and len(tc) >= 1:
        gmk = f"{group}_{cond}"
        if gmk in rgm.group_channel_medians:
            ch_times, ch_meds = rgm.group_channel_medians[gmk]
            if len(ch_times) >= 1:
                # 个体在最后一个训练时间点的通道值
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

                    # 组级通道在最后训练时间点的dE
                    closest_t = ch_times[np.argmin(np.abs(np.array(ch_times) - tc[-1]))]
                    grp_ch = ch_meds[closest_t]
                    grp_dE = np.sqrt(
                        grp_ch["dL"] ** 2 + grp_ch["da"] ** 2 + grp_ch["db"] ** 2
                    )

                    if grp_dE > 0.01:
                        scale = ind_dE / grp_dE
                        p_ind_ch_scaled = p_ch * scale

    # ---- 策略8: cross_sample - 组内跨样本平均预测 ----
    p_cross = None
    # 对同组其他样本分别预测，取中位数
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

    # ---- 策略9: saturation - 用饱和模型预测 ----
    p_sat = rgm.predict_group_dE_saturation(group, cond, t_pred)
    # 也尝试个体饱和模型
    if len(tc) >= 3:
        ind_sat = best_saturation_model(tc, dEc, min_score=-2.0)
        if ind_sat:
            p_ind_sat = predict_model(ind_sat, t_pred)
            if p_ind_sat is not None and np.isfinite(p_ind_sat):
                if p_sat is not None:
                    # 平均组级和个体饱和模型
                    p_sat = (p_sat + p_ind_sat) / 2.0
                else:
                    p_sat = p_ind_sat

    # ---- 策略10: conservative - 加权偏低的保守预测 ----
    p_conservative = None
    all_valid = []
    for p in [p_ind, p_grp, p_ch, p_scaled, p_sat, p_cross]:
        if p is not None and np.isfinite(p) and p > 0:
            all_valid.append(p)
    if all_valid:
        # 取所有预测的几何均值（偏向较低值）
        p_conservative = np.exp(np.mean(np.log(all_valid)))

    # 应用物理上界约束
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
    }


# ===================== LOLO 预计算 =====================
def precompute_lolo_strategies(df_train):
    """对每个LOLO评估点，计算所有10种策略的预测值"""
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


# ===================== 权重评估和搜索 =====================
def eval_weights_weighted(lolo_df, weights_dict):
    """测试集加权MAE评估"""
    weighted_errors = []
    group_errors = {}
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = weights_dict.get(
            g,
            weights_dict.get("default", {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES})
        )
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
    """简单平均MAE评估"""
    all_errors = []
    group_errors = {}
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = weights_dict.get(
            g,
            weights_dict.get("default", {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES})
        )
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


def search_best_weights(lolo_df, n_trials=50000, eval_fn=None):
    """只对测试集组搜索最优权重，跳过other组"""
    if eval_fn is None:
        eval_fn = eval_weights_weighted
    np.random.seed(42)
    best_weights = {}

    # 只优化测试集中的组
    for group in TEST_GROUPS:
        gdf = lolo_df[lolo_df["group"] == group]
        if len(gdf) < 2:
            best_weights[group] = {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}
            continue

        best_mae = float("inf")
        best_w = None

        for trial in range(n_trials):
            # Dirichlet采样 + 允许部分权重为0
            raw_w = np.random.dirichlet(np.ones(len(STRATEGY_NAMES)) * 0.3)
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
            uw_mae, _ = eval_weights_unweighted(gdf, {group: best_weights[group]})
            top_strats = sorted(
                zip(STRATEGY_NAMES, best_w), key=lambda x: -x[1]
            )[:4]
            top_str = {k: round(v, 3) for k, v in top_strats if v > 0.01}
            print(f"  {group:15s}: wMAE={best_mae:.4f}, uMAE={uw_mae:.4f}, top={top_str}")
        else:
            best_weights[group] = {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}

    # other组用均匀权重（不在测试集中）
    best_weights["other"] = {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES}

    return best_weights


# ===================== 预测函数 =====================
def make_prediction(rgm, sample, cond, t_pred, df_train, sample_models, weights):
    """用给定的权重做单个预测"""
    strats = compute_strategies(rgm, sample, cond, t_pred, df_train, sample_models)
    if strats is None: return 0.0

    group = detect_group(sample)
    w = weights.get(group, {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES})
    ws, wp = 0, 0
    for sn in STRATEGY_NAMES:
        v = strats[sn]
        if v is not None:
            ws += w.get(sn, 0)
            wp += w.get(sn, 0) * v
    return wp / ws if ws > 0 else 0.0


def compute_r2_and_pred(lolo_df, weights):
    """计算R²和预测值"""
    all_true, all_pred = [], []
    for _, row in lolo_df.iterrows():
        g = row["group"]
        w = weights.get(g, {sn: 1 / len(STRATEGY_NAMES) for sn in STRATEGY_NAMES})
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
    print("  颜料老化色差预测 v36")
    print("  10策略 + 饱和外推 + 跨样本迁移 + 物理约束 + 5万次权重搜索")
    print("  测试集感知优化（不优化other组）")
    print("=" * 70)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 打印数据概览
    print("\n[训练数据概览]")
    for group in TEST_GROUPS:
        sub = df_train[df_train["sample"].apply(detect_group) == group]
        samples = sub["sample"].unique()
        times = sorted(sub[sub["aging_time_day"] > 0]["aging_time_day"].unique())
        print(f"  {group:15s}: {len(samples)} 样本, 时间={times}")
    other = df_train[df_train["sample"].apply(detect_group) == "other"]
    print(f"  {'other':15s}: {len(other['sample'].unique())} 样本 (不在测试集)")

    # ====== 步骤1: LOLO预计算 ======
    print("\n[步骤1] LOLO策略预计算 (10策略)...")
    lolo_df = precompute_lolo_strategies(df_train)
    print(f"  共 {len(lolo_df)} 个评估点")

    # 各组评估点统计
    for g in TEST_GROUPS + ["other"]:
        n = len(lolo_df[lolo_df["group"] == g])
        if n > 0:
            print(f"    {g:15s}: {n} 个LOLO点")

    # ====== 步骤2: 搜索最优权重 ======
    print(f"\n[步骤2] 搜索最优权重 (50000次/组, 只优化测试集组)...")
    best_weights = search_best_weights(lolo_df, n_trials=50000)

    # ====== 步骤3: 评估最优权重 ======
    print("\n[步骤3] LOLO评估结果:")
    opt_wmae, opt_ge = eval_weights_weighted(lolo_df, best_weights)
    opt_umae, _ = eval_weights_unweighted(lolo_df, best_weights)

    all_true, all_pred = compute_r2_and_pred(lolo_df, best_weights)
    r2 = r2_score(all_true, all_pred)

    print(f"\n  整体评估 (所有LOLO点):")
    print(f"    测试集加权MAE = {opt_wmae:.4f}")
    print(f"    简单平均MAE   = {opt_umae:.4f}")
    print(f"    R2            = {r2:.4f}")

    print(f"\n  各组MAE:")
    for g in TEST_GROUPS + ["other"]:
        errs = opt_ge.get(g, [])
        if len(errs) >= 1:
            print(f"    {g:15s}: MAE={np.mean(errs):.4f} (n={len(errs)})")

    # ====== 步骤4: 生成测试集预测 ======
    print("\n[步骤4] 测试集预测...")
    rgm = RobustGroupModel(df_train)

    # 预计算个体模型
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

    # ====== 步骤5: 保存预测 ======
    print("\n[步骤5] 保存预测结果...")
    pd.DataFrame({TARGET: test_preds}).to_csv(DATA_DIR / "predict_out.csv", index=False)
    pd.DataFrame({TARGET: test_preds}).to_csv(SCRIPT_DIR / "predict_out_v36.csv", index=False)
    print(f"  已保存: {DATA_DIR / 'predict_out.csv'}")
    print(f"  已保存: {SCRIPT_DIR / 'predict_out_v36.csv'}")

    # 打印预测明细
    print(f"\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        g = detect_group(row["sample"])
        print(
            f"  {row['sample']:20s} ({row['aging_condition']:12s}, "
            f"t={row['aging_time_day']:3.0f}d) [{g:11s}] -> {test_preds[i]:.4f}"
        )

    # ====== 步骤6: 生成ensemble文件 ======
    print("\n[步骤6] 生成ensemble文件...")

    # v14 ensemble
    v14_path = SCRIPT_DIR / "predict_out_v14.csv"
    if v14_path.exists():
        v14 = pd.read_csv(v14_path)[TARGET].values
        print(f"  找到v14预测 ({len(v14)} 行)")
        for w36, label in [(0.3, "0.3v14_0.7v36"), (0.5, "0.5v14_0.5v36"), (0.7, "0.7v14_0.3v36")]:
            ens = w36 * v14 + (1 - w36) * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_{label}.csv", index=False)
        print(f"  [v14+v36 ensemble] 已生成 3 个文件")

    # v34 ensemble
    v34_path = SCRIPT_DIR / "predict_out_v34.csv"
    if v34_path.exists():
        v34 = pd.read_csv(v34_path)[TARGET].values
        print(f"  找到v34预测 ({len(v34)} 行)")
        for w36, label in [(0.3, "0.3v34_0.7v36"), (0.5, "0.5v34_0.5v36"), (0.7, "0.7v34_0.3v36")]:
            ens = w36 * v34 + (1 - w36) * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_{label}.csv", index=False)
        print(f"  [v34+v36 ensemble] 已生成 3 个文件")

    # 3-model ensemble
    if v14_path.exists() and v34_path.exists():
        v14 = pd.read_csv(v14_path)[TARGET].values
        v34 = pd.read_csv(v34_path)[TARGET].values
        for w14, w34, label in [
            (0.2, 0.3, "0.2v14_0.3v34_0.5v36"),
            (0.3, 0.3, "0.3v14_0.3v34_0.4v36"),
            (0.25, 0.25, "0.25v14_0.25v34_0.5v36"),
        ]:
            ens = w14 * v14 + w34 * v34 + (1 - w14 - w34) * test_preds
            pd.DataFrame({TARGET: ens}).to_csv(SCRIPT_DIR / f"predict_out_3ens_{label}.csv", index=False)
        print(f"  [3-model ensemble] 已生成 3 个文件")

    # ====== 最终总结 ======
    print("\n" + "=" * 70)
    print("  最终结果")
    print("=" * 70)
    print(f"\n  v36 LOLO评估:")
    print(f"    R2  = {r2:.4f}")
    print(f"    MAE = {opt_umae:.4f}")
    print(f"    wMAE= {opt_wmae:.4f}")
    print(f"\n  各组权重:")
    for g in TEST_GROUPS:
        w = best_weights[g]
        top = sorted(w.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}={v:.3f}" for k, v in top if v > 0.01)
        print(f"    {g:15s}: {top_str}")
    print(f"\n  历史对比:")
    print(f"    v34: R2=0.967, MAE=0.340")
    print(f"    v32: R2=0.960, MAE=0.375")
    print(f"    v28: R2=0.918, MAE=0.626")


if __name__ == "__main__":
    main()
