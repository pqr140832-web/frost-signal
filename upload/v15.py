"""
颜料老化色差预测 - v15
基于Claude Opus 4.7建议的改进版本

Claude Opus的关键建议：
1. 🔴 钴蓝/翡翠绿用Avrami饱和方程 A(1-exp(-k*t^n))，幂律会外推过头
2. 🔴 曙红固定n用有机染料组均值，只拟合A（2个点不能拟合2个参数）
3. 🟡 双路集成：直接dE建模 + 通道分解，按组选权重
4. 🟡 染料组做主导通道分析，只精细建模主导通道
5. 🟢 保守预测：数据少时向组均值收缩
6. 🟢 诊断：dE(24)/dE(18) vs dE(18)/dE(12)比值下降→饱和型
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== 路径配置 =====================
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

# ===================== 2. 数据准备 =====================
def prepare_series(df_sub: pd.DataFrame) -> tuple:
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)

def prepare_channels(df_sub: pd.DataFrame) -> tuple:
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
    if len(t) < 3:
        return t.copy(), y.copy()
    diffs = np.diff(y)
    mean_abs = np.mean(np.abs(diffs))
    if mean_abs < 1e-6:
        return t, y
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t) - 1):
        if abs(diffs[i - 1]) > threshold * mean_abs:
            keep[i] = False
    return t[keep], y[keep]

# ===================== 3. 生长模型库 =====================
def fit_power_law(t, y):
    """dE = A * t^n"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 2:
        return None
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

def fit_logarithmic(t, y):
    """dE = A * log(1 + k*t)"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 2:
        return None
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

def fit_avrami(t, y):
    """Avrami方程: dE = A * (1 - exp(-k * t^n))
    适用于饱和型老化（无机颜料如钴蓝、翡翠绿）"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    t, y = t[mask], y[mask]
    y_max = y.max()
    t_max = t.max()
    if y_max < 1e-6 or t_max < 1e-6:
        return None
    try:
        def avrami(tn, A, k, n):
            return A * (1 - np.exp(-k * np.power(tn, n)))
        # 多组初始值尝试
        best_fit = None
        best_score = -1e9
        for A0 in [y_max * 1.2, y_max * 1.5, y_max * 2.0]:
            for n0 in [0.3, 0.5, 0.8, 1.0]:
                for k0 in [0.001, 0.01, 0.1]:
                    try:
                        popt, _ = curve_fit(
                            avrami, t, y,
                            p0=[A0, k0, n0],
                            bounds=([0, 1e-8, 0.05], [A0 * 5, 1.0, 3.0]),
                            maxfev=3000
                        )
                        pred = avrami(t, *popt)
                        ss_res = np.sum((y - pred) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2)
                        score = 1 - ss_res / (ss_tot + 1e-9)
                        if score > best_score and popt[0] > 0 and popt[1] > 0:
                            best_score = score
                            best_fit = popt
                    except:
                        pass
        if best_fit is not None:
            return {"type": "avrami", "A": best_fit[0], "k": best_fit[1], "n": best_fit[2], "score": best_score}
    except:
        pass
    return None

def fit_michaelis_menten(t, y):
    """dE = A * t / (B + t)"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    t, y = t[mask], y[mask]
    try:
        t_max, y_max = t.max(), y.max()
        t_norm, y_norm = t / t_max, y / y_max
        def mm(tn, A, B):
            return A * tn / (B + tn)
        popt, _ = curve_fit(mm, t_norm, y_norm, p0=[1.0, 0.5], maxfev=5000)
        A, B = popt
        pred = y_max * mm(t_norm, A, B)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "mm", "A": A * y_max, "B": B * t_max, "score": score}
    except:
        return None

def fit_linear(t, y):
    """dE = a + b*t"""
    if len(t) < 2:
        return None
    A = np.vstack([np.ones_like(t), t]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "linear", "a": coeffs[0], "b": coeffs[1], "score": score}
    except:
        return None

def predict_growth(model, t_pred):
    if model is None:
        return None
    if model["type"] == "power":
        return max(model["A"] * (t_pred ** model["n"]), 0)
    elif model["type"] == "log":
        return model["A"] * np.log(1 + model["k"] * t_pred)
    elif model["type"] == "avrami":
        return max(model["A"] * (1 - np.exp(-model["k"] * (t_pred ** model["n"]))), 0)
    elif model["type"] == "mm":
        return model["A"] * t_pred / (model["B"] + t_pred)
    elif model["type"] == "linear":
        return max(model["a"] + model["b"] * t_pred, 0)
    return None

def best_growth_model(t, y, include_avrami=False):
    models = [fit_power_law(t, y), fit_logarithmic(t, y), fit_michaelis_menten(t, y), fit_linear(t, y)]
    if include_avrami:
        models.append(fit_avrami(t, y))
    models = [m for m in models if m is not None and m.get("score", -999) > -1.0]
    if not models:
        return None
    return max(models, key=lambda m: m["score"])

# ===================== 4. 改进的分层模型(Avrami+幂律+保守收缩) =====================
class ImprovedHierarchicalModel:
    """
    改进的分层模型：
    - 钴蓝/翡翠绿用Avrami饱和方程
    - 曙红固定n用有机染料组均值
    - 保守预测收缩策略
    """

    def __init__(self, df_train: pd.DataFrame):
        self.models = {}
        self.organic_n_avg = None  # 有机染料组的平均n
        self._build_all(df_train)

    def _build_all(self, df_train: pd.DataFrame):
        # Step 1: 先计算有机染料组的平均n（用于曙红先验）
        organic_n_values = []
        for sample in df_train["sample"].unique():
            group = detect_group(sample)
            if group == "dye":
                sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "UV")]
                t, dE = prepare_series(sub)
                if len(t) >= 3:
                    model = fit_power_law(t, dE)
                    if model and 0.1 < model["n"] < 2.0:
                        organic_n_values.append(model["n"])
        if organic_n_values:
            self.organic_n_avg = np.median(organic_n_values)

        # Step 2: 按组构建模型
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            all_t, all_dE = [], []
            for m in members:
                sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == "UV")].sort_values("aging_time_day")
                t, dE = prepare_series(sub)
                if len(t) >= 2:
                    all_t.extend(t)
                    all_dE.extend(dE)

            if len(all_t) < 4:
                continue

            all_t = np.array(all_t)
            all_dE = np.array(all_dE)

            # 钴蓝和翡翠绿：Avrami + 幂律，选最佳
            if group in ["jade_green", "cobalt_blue"]:
                model_avrami = fit_avrami(all_t, all_dE)
                model_power = fit_power_law(all_t, all_dE)
                model_log = fit_logarithmic(all_t, all_dE)

                candidates = [m for m in [model_avrami, model_power, model_log] if m is not None and m.get("score", -1) > -1]
                if candidates:
                    best = max(candidates, key=lambda m: m["score"])
                    self.models[group] = {**best, "members": members}

                # 也存储幂律n作为备用
                if model_power:
                    self.models[group + "_power_n"] = model_power["n"]

            # 曙红：固定n用有机染料均值
            elif group == "shu_red":
                if self.organic_n_avg:
                    n_fixed = self.organic_n_avg
                else:
                    n_fixed = 0.6  # 默认有机染料n

                # 只拟合A
                tn = np.power(all_t, n_fixed)
                A = np.dot(tn, all_dE) / (np.dot(tn, tn) + 1e-9)
                pred = A * tn
                ss_res = np.sum((all_dE - pred) ** 2)
                ss_tot = np.sum((all_dE - all_dE.mean()) ** 2)
                score = 1 - ss_res / (ss_tot + 1e-9)

                self.models[group] = {
                    "type": "power",
                    "A": A,
                    "n": n_fixed,
                    "score": score,
                    "n_fixed": True,
                    "members": members,
                }

            # 其他组：标准幂律
            else:
                def neg_r2(n, at=all_t, ad=all_dE):
                    mask = (at > 0) & (ad > 0)
                    if mask.sum() < 2:
                        return 1e9
                    tn = np.power(at[mask], n)
                    dm = ad[mask]
                    A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
                    pred = A * tn
                    ss_res = np.sum((dm - pred) ** 2)
                    ss_tot = np.sum((dm - dm.mean()) ** 2)
                    return -(1 - ss_res / (ss_tot + 1e-9))
                result = minimize_scalar(neg_r2, bounds=(0.1, 2.0), method="bounded")
                best_n = result.x
                mask = (all_t > 0) & (all_dE > 0)
                tn = np.power(all_t[mask], best_n)
                dm = all_dE[mask]
                A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
                self.models[group] = {"type": "power", "A": A, "n": best_n, "score": -result.fun, "members": members}

    def predict(self, group, t_train, dE_train, t_pred):
        if group not in self.models:
            return None
        model = self.models[group]
        n_members = len(model.get("members", []))

        # 组级预测
        if model["type"] == "avrami":
            group_pred = model["A"] * (1 - np.exp(-model["k"] * (t_pred ** model["n"])))
        else:
            group_pred = model["A"] * (t_pred ** model["n"])

        group_pred = max(group_pred, 0)

        # 个体缩放因子
        tc, dEc = remove_outliers(t_train, dE_train)
        if len(tc) == 0:
            return group_pred

        # 对Avrami模型，缩放因子基于模型预测值
        ratios = []
        for i, t in enumerate(tc):
            if t > 0 and dEc[i] > 0:
                if model["type"] == "avrami":
                    g_at_t = model["A"] * (1 - np.exp(-model["k"] * (t ** model["n"])))
                else:
                    g_at_t = model["A"] * (t ** model["n"])
                if g_at_t > 1e-6:
                    ratios.append(dEc[i] / g_at_t)

        if not ratios:
            return group_pred

        # 指数衰减加权
        weights = np.array([0.7 ** (len(ratios) - 1 - i) for i in range(len(ratios))])
        weights /= weights.sum()
        scale = np.average(ratios, weights=weights)

        individual_pred = group_pred * scale

        # ★ 保守预测收缩（Claude Opus建议）
        # λ按数据点数调整：2点λ=0.6, 3点λ=0.8, 4点以上λ=0.95
        n_points = len(tc)
        if n_points <= 2:
            lam = 0.55  # 极保守
        elif n_points == 3:
            lam = 0.75
        else:
            lam = 0.92

        # 噪声大时更保守
        if len(dEc) >= 2:
            noise = np.std(dEc) / max(np.mean(dEc), 1e-6)
            if noise > 0.5:
                lam *= 0.85
            elif noise > 0.3:
                lam *= 0.92

        # 组成员多→组模型更可信→更保守
        if n_members >= 7:
            lam *= 0.95

        final = lam * individual_pred + (1 - lam) * group_pred
        return float(max(final, 0))

# ===================== 5. 颜色通道分解模型(改进版) =====================
class ChannelModelV2:
    """
    改进的通道分解模型：
    - 对主导通道精细建模，次要通道用简单模型
    - 用中位数组统计抗噪声
    """

    def __init__(self, df_train: pd.DataFrame):
        self.group_models = {}
        self.group_stats = {}
        self.dominant_channels = {}  # 每组的主导通道
        self._build_all(df_train)

    def _build_all(self, df_train: pd.DataFrame):
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            group_models = {}
            group_stats = {}
            ch_contributions = {"dL": 0, "da": 0, "db": 0}

            for ch_idx, ch_name in enumerate(["dL", "da", "db"]):
                all_t, all_ch = [], []
                for m in members:
                    sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == "UV")].sort_values("aging_time_day")
                    t, dL, da, db = prepare_channels(sub)
                    ch_data = [dL, da, db][ch_idx]
                    if len(t) >= 2:
                        all_t.extend(t)
                        all_ch.extend(ch_data)
                        ch_contributions[ch_name] += np.mean(np.abs(ch_data))

                if len(all_t) < 3:
                    continue

                all_t = np.array(all_t)
                all_ch = np.array(all_ch)

                # 线性模型（允许负数）
                linear = fit_linear(all_t, all_ch)

                # 组级通道统计（中位数）
                unique_times = np.unique(all_t)
                ch_means = []
                for ut in unique_times:
                    mask = np.isclose(all_t, ut)
                    ch_means.append(np.median(all_ch[mask]))
                group_stats[ch_name] = {
                    "times": unique_times,
                    "means": np.array(ch_means),
                    "std": np.std(all_ch),
                }

                group_models[ch_name] = {"linear_model": linear}

            self.group_models[group] = group_models
            self.group_stats[group] = group_stats

            # 确定主导通道
            total = sum(ch_contributions.values())
            if total > 0:
                self.dominant_channels[group] = max(ch_contributions, key=ch_contributions.get)

    def predict(self, group, t_train, dE_train, sample, df_train, t_pred, cond="UV"):
        sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)].sort_values("aging_time_day")
        if len(sub) == 0:
            return None
        t_ch, dL, da, db = prepare_channels(sub)
        if len(t_ch) == 0:
            return None

        dominant = self.dominant_channels.get(group, "db")
        preds = {}

        for ch_name, ch_data in [("dL", dL), ("da", da), ("db", db)]:
            stats = self.group_stats.get(group, {}).get(ch_name)
            gmodel = self.group_models.get(group, {}).get(ch_name)

            # ★ 主导通道：精细建模（个体缩放+组级趋势）
            if ch_name == dominant and stats is not None:
                # 计算个体缩放因子
                if len(t_ch) >= 1 and len(stats["times"]) > 0:
                    best_t_idx = np.argmin(np.abs(stats["times"] - t_ch[-1]))
                    group_mean_at_t = stats["means"][best_t_idx]
                    if abs(group_mean_at_t) > 1e-6:
                        scale = ch_data[-1] / group_mean_at_t

                        # 组级趋势外推（线性）
                        if gmodel and gmodel["linear_model"]:
                            p_linear = predict_growth(gmodel["linear_model"], t_pred)
                            preds[ch_name] = p_linear * scale
                        else:
                            # 用组级平均增长率
                            if len(stats["times"]) >= 2:
                                group_rate = (stats["means"][-1] - stats["means"][-2]) / (stats["times"][-1] - stats["times"][-2])
                                group_pred = stats["means"][-1] + group_rate * (t_pred - stats["times"][-1])
                                preds[ch_name] = group_pred * scale
                            else:
                                preds[ch_name] = ch_data[-1]
                        continue

            # ★ 次要通道：简单线性外推
            if len(ch_data) >= 2:
                t1, t2 = t_ch[-2], t_ch[-1]
                d1, d2 = ch_data[-2], ch_data[-1]
                if t2 > t1:
                    rate = (d2 - d1) / (t2 - t1)
                    preds[ch_name] = d2 + rate * (t_pred - t2)
                else:
                    preds[ch_name] = d2
            elif len(ch_data) >= 1:
                preds[ch_name] = ch_data[-1]

        if len(preds) != 3:
            return None

        dE = np.sqrt(preds["dL"]**2 + preds["da"]**2 + preds["db"]**2)
        return float(max(dE, 0))

# ===================== 6. humid-heat独立建模 =====================
class HumidHeatModel:
    def __init__(self, df_train: pd.DataFrame):
        self.hh_models = {}
        self._build(df_train)

    def _build(self, df_train: pd.DataFrame):
        for sample in df_train["sample"].unique():
            hh = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "humid-_heat")]
            if len(hh) < 3:
                continue
            t_hh, dE_hh = prepare_series(hh)
            if len(t_hh) < 2:
                continue
            model_hh = best_growth_model(t_hh, dE_hh)
            self.hh_models[sample] = model_hh

    def predict(self, sample, cond, t_train, dE_train, t_pred):
        if cond != "humid-_heat":
            return None
        if sample in self.hh_models and self.hh_models[sample] is not None:
            return predict_growth(self.hh_models[sample], t_pred)
        tc, dEc = remove_outliers(t_train, dE_train)
        if len(tc) >= 2:
            model = best_growth_model(tc, dEc)
            if model:
                return predict_growth(model, t_pred)
        return None

# ===================== 7. v15集成 =====================
class V15Ensemble:
    def __init__(self, df_train: pd.DataFrame):
        self.hier = ImprovedHierarchicalModel(df_train)
        self.channel = ChannelModelV2(df_train)
        self.hh_model = HumidHeatModel(df_train)
        self.df_train = df_train

    def predict_single(self, sample, cond, t_pred):
        group = detect_group(sample)

        sub = self.df_train[
            (self.df_train["sample"] == sample) &
            (self.df_train["aging_condition"] == cond)
        ].sort_values("aging_time_day")

        if len(sub) == 0:
            return float(self.df_train[TARGET].mean())

        t_arr, dE_arr = prepare_series(sub)
        if len(t_arr) == 0:
            return 0.0

        # humid-heat特殊处理
        if cond == "humid-_heat":
            p_hh = self.hh_model.predict(sample, cond, t_arr, dE_arr, t_pred)
            if p_hh is not None:
                return p_hh

        # 双路预测
        p_hier = self.hier.predict(group, t_arr, dE_arr, t_pred)
        p_ch = self.channel.predict(group, t_arr, dE_arr, sample, self.df_train, t_pred, cond)

        predictions = {}
        if p_hier is not None:
            predictions["hier"] = p_hier
        if p_ch is not None:
            predictions["channel"] = p_ch

        if not predictions:
            return 0.0

        # ★ 按组选集成策略
        return self._ensemble(group, predictions, t_arr, dE_arr)

    def _ensemble(self, group, predictions, t_train, dE_train):
        n_points = len(t_train)

        # 染料组：分层模型为主
        if group == "dye":
            if "hier" in predictions:
                return predictions["hier"]
            return predictions.get("channel", 0)

        # 翡翠绿和钴蓝：Avrami分层 + 通道平均
        if group in ["jade_green", "cobalt_blue"]:
            preds = list(predictions.values())
            return np.mean(preds)

        # 曙红：2个数据点，保守策略
        if group == "shu_red":
            if n_points <= 2:
                if len(predictions) == 2:
                    # 两个模型平均（更保守）
                    return np.mean(list(predictions.values()))
                return list(predictions.values())[0]
            else:
                if "hier" in predictions:
                    return predictions["hier"]
                return predictions.get("channel", 0)

        # 皮纸
        if group == "paper":
            if "hier" in predictions:
                return predictions["hier"]
            return predictions.get("channel", 0)

        return np.mean(list(predictions.values()))

# ===================== 8. 前向评估 =====================
def forward_eval(df_train, ensemble):
    y_true, y_pred = [], []
    details = []

    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond)
            ].sort_values("aging_time_day")

            if len(sub) < 3:
                continue

            train_sub = sub.iloc[:-1]
            test_row = sub.iloc[-1]

            t_arr, dE_arr = prepare_series(train_sub)
            if len(t_arr) == 0:
                continue

            t_tgt = float(test_row["aging_time_day"])
            pred = ensemble.predict_single(sample, cond, t_tgt)

            y_true.append(float(test_row[TARGET]))
            y_pred.append(pred)
            details.append({
                "sample": sample,
                "group": detect_group(sample),
                "cond": cond,
                "t": t_tgt,
                "y_true": float(test_row[TARGET]),
                "y_pred": pred,
            })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "n": len(y_true),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "details": details,
    }

# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("  颜料老化色差预测 v15")
    print("  Claude Opus建议: Avrami+保守收缩+主导通道")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 构建模型
    print("\n[模型] 构建v15集成模型...")
    ensemble = V15Ensemble(df_train)

    # 打印模型参数
    print(f"\n[分层模型] 有机染料组平均n={ensemble.hier.organic_n_avg:.3f}" if ensemble.hier.organic_n_avg else "")
    print("\n[分层模型] 组级参数:")
    for g, m in ensemble.hier.models.items():
        if "power_n" in g:
            continue
        n_fixed = m.get("n_fixed", False)
        print(f"  {g:15s}: type={m['type']:8s}, n={m['n']:.3f}{' (fixed)' if n_fixed else ''}, A={m['A']:.4f}, score={m['score']:.4f}")

    # 打印主导通道
    print("\n[通道] 主导通道:")
    for g, ch in ensemble.channel.dominant_channels.items():
        print(f"  {g:15s}: {ch}")

    # 前向验证
    print("\n[评估] 前向外推评估...")
    metrics = forward_eval(df_train, ensemble)
    print(f"  有效序列数: {metrics['n']}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")

    # 按组评估
    print("\n[评估] 按组详细:")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
        gdetails = [d for d in metrics["details"] if d["group"] == g]
        if len(gdetails) >= 2:
            yt = np.array([d["y_true"] for d in gdetails])
            yp = np.array([d["y_pred"] for d in gdetails])
            r2 = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            print(f"  {g:15s}: R²={r2:+.4f}, MAE={mae:.4f}, n={len(gdetails)}")

    # 测试集预测
    print("\n[预测] 生成测试集预测...")
    test_preds = []
    for _, row in df_test.iterrows():
        pred = ensemble.predict_single(row["sample"], row["aging_condition"], float(row["aging_time_day"]))
        test_preds.append(pred)

    test_preds = np.array(test_preds, dtype=float)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d)"
              f"  →  {test_preds[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"  v15:  R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    print(f"  v14:  R²=0.9879, MAE=0.1975, RMSE=0.2985")
    print(f"  v13:  R²=0.9784, MAE=0.2906")
    print("=" * 60)


if __name__ == "__main__":
    main()
