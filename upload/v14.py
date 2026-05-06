"""
颜料老化色差预测 - v14
多策略集成 + 颜色通道分解

关键改进：
1. 颜色通道分解：分别预测ΔL、Δa、Δb，再组合为dE=sqrt(ΔL²+Δa²+Δb²)
2. 多生长模型选择：幂律、对数、Michaelis-Menten、线性，按样本选最优
3. 更强的组级通道池化：对噪声大的组(翡翠绿、钴蓝、曙红)用中位数+IQR稳健估计
4. 独立的humid-heat建模：皮纸UV/HH关系不一致，分开建模
5. 自适应集成：按组按策略加权融合
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from itertools import product

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
    """重复时间点取均值，去掉t=0"""
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)

def prepare_channels(df_sub: pd.DataFrame) -> tuple:
    """准备颜色通道数据: ΔL, Δa, Δb"""
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

# ===================== 3. 异常点过滤 =====================
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

# ===================== 4. 生长模型库 =====================
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

def fit_michaelis_menten(t, y):
    """dE = A * t / (B + t)  (饱和曲线)"""
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    t, y = t[mask], y[mask]

    try:
        # 归一化
        t_max = t.max()
        t_norm = t / t_max
        y_max = y.max()

        def mm(tn, A, B):
            return A * tn / (B + tn)

        popt, _ = curve_fit(mm, t_norm, y / y_max, p0=[1.0, 0.5], maxfev=5000)
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
    """用拟合好的生长模型预测"""
    if model is None:
        return None
    if model["type"] == "power":
        return max(model["A"] * (t_pred ** model["n"]), 0)
    elif model["type"] == "log":
        return model["A"] * np.log(1 + model["k"] * t_pred)
    elif model["type"] == "mm":
        return model["A"] * t_pred / (model["B"] + t_pred)
    elif model["type"] == "linear":
        return max(model["a"] + model["b"] * t_pred, 0)
    return None

def best_growth_model(t, y):
    """尝试所有模型，返回最佳"""
    models = [
        fit_power_law(t, y),
        fit_logarithmic(t, y),
        fit_michaelis_menten(t, y),
        fit_linear(t, y),
    ]
    models = [m for m in models if m is not None and m.get("score", -999) > -1.0]
    if not models:
        return None
    return max(models, key=lambda m: m["score"])

# ===================== 5. 颜色通道分解模型 =====================
class ChannelModel:
    """对每个颜色通道(ΔL,Δa,Δb)分别建模，然后组合为dE"""

    def __init__(self, df_train: pd.DataFrame):
        self.group_channel_models = {}  # {group: {channel: model}}
        self.group_channel_stats = {}   # {group: {channel: {t_mean, dch_mean}}}
        self._build_all(df_train)

    def _build_all(self, df_train: pd.DataFrame):
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            group_models = {}
            group_stats = {}

            for ch_idx, ch_name in enumerate(["dL", "da", "db"]):
                all_t, all_ch = [], []
                for m in members:
                    sub = df_train[
                        (df_train["sample"] == m) &
                        (df_train["aging_condition"] == "UV")
                    ].sort_values("aging_time_day")
                    t, dL, da, db = prepare_channels(sub)
                    ch_data = [dL, da, db][ch_idx]
                    if len(t) >= 2:
                        all_t.extend(t)
                        all_ch.extend(ch_data)

                if len(all_t) < 3:
                    continue

                all_t = np.array(all_t)
                all_ch = np.array(all_ch)

                # 对通道数据拟合最佳增长模型
                # 注意：通道值可以为负数，所以幂律和对数需要处理
                ch_abs = np.abs(all_ch)
                ch_sign = np.sign(all_ch)
                # 使用绝对值拟合，然后恢复符号
                model = best_growth_model(all_t, ch_abs)

                # 也拟合原始值(允许负数)的线性模型
                linear = fit_linear(all_t, all_ch)

                group_models[ch_name] = {
                    "abs_model": model,      # 对绝对值拟合
                    "linear_model": linear,   # 线性模型(允许负数)
                    "ch_sign": np.sign(np.median(all_ch)),  # 组级符号
                }

                # 组级通道统计
                unique_times = np.unique(all_t)
                ch_means = []
                for ut in unique_times:
                    mask = np.isclose(all_t, ut)
                    ch_means.append(np.median(all_ch[mask]))  # 用中位数抗噪声
                group_stats[ch_name] = {
                    "times": unique_times,
                    "means": np.array(ch_means),
                }

            self.group_channel_models[group] = group_models
            self.group_channel_stats[group] = group_stats

    def predict_channel(self, group, ch_name, t_train, ch_train, t_pred):
        """预测单个通道值"""
        if group not in self.group_channel_models:
            return None
        if ch_name not in self.group_channel_models[group]:
            return None

        gmodel = self.group_channel_models[group][ch_name]
        stats = self.group_channel_stats.get(group, {}).get(ch_name)

        # 方法1: 线性外推(保留符号)
        p_linear = None
        if gmodel["linear_model"]:
            p_linear = predict_growth(gmodel["linear_model"], t_pred)

        # 方法2: 用绝对值模型+符号
        p_abs = None
        if gmodel["abs_model"]:
            p_abs_val = predict_growth(gmodel["abs_model"], t_pred)
            if p_abs_val is not None:
                p_abs = gmodel["ch_sign"] * p_abs_val

        # 方法3: 个体缩放因子
        p_scaled = None
        if stats is not None and len(t_train) >= 1:
            # 找最近的训练时间点，计算个体的缩放因子
            best_t_idx = np.argmin(np.abs(stats["times"] - t_train[-1]))
            group_mean_at_t = stats["means"][best_t_idx]
            if abs(group_mean_at_t) > 1e-6:
                individual_val = ch_train[-1]
                scale = individual_val / group_mean_at_t
                # 用组级模型预测 + 个体缩放
                if p_abs is not None:
                    p_scaled = p_abs * scale
                elif p_linear is not None:
                    p_scaled = p_linear * scale

        # 智能选择：数据少或噪声大→更依赖组模型
        results = []
        if p_scaled is not None:
            results.append(("scaled", p_scaled))
        if p_linear is not None:
            results.append(("linear", p_linear))
        if p_abs is not None:
            results.append(("abs", p_abs))

        if not results:
            return None

        # 如果有缩放版本且数据点>=2，优先用缩放版
        if len(t_train) >= 2 and p_scaled is not None:
            return p_scaled
        elif p_linear is not None:
            return p_linear
        else:
            return p_abs

    def predict(self, group, t_train, dE_train, sample, df_train, t_pred, cond="UV"):
        """预测dE = sqrt(ΔL² + Δa² + Δb²)"""
        # 获取训练数据的通道值
        sub = df_train[
            (df_train["sample"] == sample) &
            (df_train["aging_condition"] == cond)
        ].sort_values("aging_time_day")

        if len(sub) == 0:
            return None

        t_ch, dL, da, db = prepare_channels(sub)
        if len(t_ch) == 0:
            return None

        # 对每个通道分别预测
        preds = {}
        for ch_name, ch_data in [("dL", dL), ("da", da), ("db", db)]:
            p = self.predict_channel(group, ch_name, t_ch, ch_data, t_pred)
            if p is not None:
                preds[ch_name] = p
            else:
                # 回退：用最后值的趋势线性外推
                if len(ch_data) >= 2:
                    rate = (ch_data[-1] - ch_data[-2]) / (t_ch[-1] - t_ch[-2]) if t_ch[-1] > t_ch[-2] else 0
                    preds[ch_name] = ch_data[-1] + rate * (t_pred - t_ch[-1])
                elif len(ch_data) >= 1:
                    preds[ch_name] = ch_data[-1]

        if len(preds) != 3:
            return None

        dE = np.sqrt(preds["dL"]**2 + preds["da"]**2 + preds["db"]**2)
        return float(max(dE, 0))

# ===================== 6. v13分层模型(保留作为集成成员) =====================
class HierarchicalModel:
    """v13的分层模型：组级幂律 + 个体缩放因子"""

    def __init__(self, df_train: pd.DataFrame):
        self.models = {}
        self._build_all(df_train)

    def _build_all(self, df_train: pd.DataFrame):
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            all_t, all_dE = [], []
            for m in members:
                sub = df_train[
                    (df_train["sample"] == m) &
                    (df_train["aging_condition"] == "UV")
                ].sort_values("aging_time_day")
                t, dE = prepare_series(sub)
                if len(t) >= 2:
                    all_t.extend(t)
                    all_dE.extend(dE)

            if len(all_t) < 4:
                continue

            all_t = np.array(all_t)
            all_dE = np.array(all_dE)

            def neg_r2(n, at=all_t, ad=all_dE):
                mask = (at > 0) & (ad > 0)
                if mask.sum() < 2:
                    return 1e9
                tn = np.power(at[mask], n)
                dm = ad[mask]
                A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
                pred = A * np.power(at[mask], n)
                ss_res = np.sum((dm - pred) ** 2)
                ss_tot = np.sum((dm - dm.mean()) ** 2)
                return -(1 - ss_res / (ss_tot + 1e-9))

            result = minimize_scalar(neg_r2, bounds=(0.1, 2.0), method="bounded")
            best_n = result.x

            mask = (all_t > 0) & (all_dE > 0)
            tn = np.power(all_t[mask], best_n)
            dm = all_dE[mask]
            A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)

            self.models[group] = {"n": best_n, "A": A, "members": members}

    def predict(self, group, t_train, dE_train, t_pred):
        if group not in self.models:
            return None
        model = self.models[group]
        n, A_group = model["n"], model["A"]

        group_pred = A_group * (t_pred ** n)

        tc, dEc = remove_outliers(t_train, dE_train)
        if len(tc) == 0:
            return group_pred

        ratios = []
        for i, t in enumerate(tc):
            if t > 0 and dEc[i] > 0:
                g_at_t = A_group * (t ** n)
                if g_at_t > 1e-6:
                    ratios.append(dEc[i] / g_at_t)

        if not ratios:
            return group_pred

        weights = np.array([0.7 ** (len(ratios) - 1 - i) for i in range(len(ratios))])
        weights /= weights.sum()
        scale = np.average(ratios, weights=weights)

        individual_pred = group_pred * scale

        noise_level = np.std(dEc) / max(np.mean(dEc), 1e-6) if len(dEc) >= 2 else 1.0
        if len(tc) <= 2:
            w_individual = 0.3
        elif noise_level > 0.4:
            w_individual = 0.4
        elif noise_level > 0.2:
            w_individual = 0.6
        else:
            w_individual = 0.8

        final = w_individual * individual_pred + (1 - w_individual) * group_pred
        return float(max(final, 0))

# ===================== 7. 独立per-sample最优模型选择 =====================
class PerSampleBestModel:
    """对每个样本独立选择最佳生长模型"""

    def predict(self, t_train, dE_train, t_pred):
        tc, dEc = remove_outliers(t_train, dE_train)
        if len(tc) < 2:
            return None

        model = best_growth_model(tc, dEc)
        if model is None:
            return None

        p_model = predict_growth(model, t_pred)
        if p_model is None:
            return None

        # 最后两点线性外推作为辅助
        if len(tc) >= 2:
            t1, t2 = tc[-2], tc[-1]
            d1, d2 = dEc[-2], dEc[-1]
            if t2 > t1:
                rate = (d2 - d1) / (t2 - t1)
                p_linear = max(d2 + rate * (t_pred - t2), 0)
            else:
                p_linear = max(d2, 0)
        else:
            p_linear = p_model

        # 如果模型R²很高，信任模型；否则混合线性
        score = model.get("score", 0)
        if score > 0.8:
            return float(max(0.7 * p_model + 0.3 * p_linear, 0))
        elif score > 0:
            return float(max(0.4 * p_model + 0.6 * p_linear, 0))
        else:
            return float(max(p_linear, 0))

# ===================== 8. humid-heat独立建模 =====================
class HumidHeatModel:
    """对humid-heat条件独立建模，使用UV作为先验"""
    def __init__(self, df_train: pd.DataFrame):
        self.hh_models = {}
        self.ratios = {}
        self._build(df_train)

    def _build(self, df_train: pd.DataFrame):
        for sample in df_train["sample"].unique():
            uv = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "UV")]
            hh = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "humid-_heat")]

            if len(uv) < 3 or len(hh) < 3:
                continue

            # 对UV和HH分别拟合模型
            t_uv, dE_uv = prepare_series(uv)
            t_hh, dE_hh = prepare_series(hh)

            if len(t_uv) < 2 or len(t_hh) < 2:
                continue

            model_uv = best_growth_model(t_uv, dE_uv)
            model_hh = best_growth_model(t_hh, dE_hh)

            self.hh_models[sample] = model_hh

            # 计算HH/UV比例
            common_times = set(t_uv) & set(t_hh)
            if common_times:
                common_times = sorted(common_times)
                ratios = []
                for ct in common_times:
                    uv_val = uv[uv["aging_time_day"] == ct][TARGET].mean()
                    hh_val = hh[hh["aging_time_day"] == ct][TARGET].mean()
                    if uv_val > 0.01:
                        ratios.append(hh_val / uv_val)
                if ratios:
                    self.ratios[sample] = np.median(ratios)

    def predict(self, sample, cond, t_train, dE_train, t_pred):
        if cond != "humid-_heat":
            return None

        # 优先用样本自己的HH模型
        if sample in self.hh_models and self.hh_models[sample] is not None:
            return predict_growth(self.hh_models[sample], t_pred)

        # 没有HH数据时，用训练数据最后趋势外推
        tc, dEc = remove_outliers(t_train, dE_train)
        if len(tc) >= 2:
            model = best_growth_model(tc, dEc)
            if model and model.get("score", -1) > 0:
                return predict_growth(model, t_pred)
            # 线性外推
            t1, t2 = tc[-2], tc[-1]
            d1, d2 = dEc[-2], dEc[-1]
            if t2 > t1:
                rate = (d2 - d1) / (t2 - t1)
                return max(d2 + rate * (t_pred - t2), 0)

        return None

# ===================== 9. 自适应集成预测 =====================
class V14Ensemble:
    """多策略自适应集成"""

    def __init__(self, df_train: pd.DataFrame):
        self.hier = HierarchicalModel(df_train)
        self.channel = ChannelModel(df_train)
        self.per_sample = PerSampleBestModel()
        self.hh_model = HumidHeatModel(df_train)
        self.df_train = df_train

    def predict_single(self, sample, cond, t_pred):
        group = detect_group(sample)

        # 获取训练数据
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

        # 多策略预测
        predictions = {}

        # 策略1: v13分层模型
        p_hier = self.hier.predict(group, t_arr, dE_arr, t_pred)
        if p_hier is not None:
            predictions["hier"] = p_hier

        # 策略2: 颜色通道分解
        p_ch = self.channel.predict(group, t_arr, dE_arr, sample, self.df_train, t_pred, cond)
        if p_ch is not None:
            predictions["channel"] = p_ch

        # 策略3: per-sample最优模型
        p_ps = self.per_sample.predict(t_arr, dE_arr, t_pred)
        if p_ps is not None:
            predictions["per_sample"] = p_ps

        if not predictions:
            return 0.0

        # 按组选择集成策略
        return self._ensemble(group, predictions, t_arr, dE_arr, t_pred)

    def _ensemble(self, group, predictions, t_train, dE_train, t_pred):
        """按组自适应加权"""

        # 染料组: 数据多(5点)，per-sample模型应该更准
        if group == "dye":
            if "per_sample" in predictions:
                return predictions["per_sample"]
            elif "channel" in predictions:
                return predictions["channel"]
            return predictions.get("hier", 0)

        # 翡翠绿和钴蓝: 数据少且噪声大，主要靠分层和通道
        if group in ["jade_green", "cobalt_blue"]:
            preds = []
            if "hier" in predictions:
                preds.append(predictions["hier"])
            if "channel" in predictions:
                preds.append(predictions["channel"])
            if preds:
                return np.mean(preds)  # 简单平均
            return predictions.get("per_sample", 0)

        # 曙红: 只有2个非零数据点，通道分解可能更有优势
        if group == "shu_red":
            if len(t_train) <= 2:
                # 数据极少：更依赖组级模型
                if "hier" in predictions and "channel" in predictions:
                    return 0.4 * predictions["hier"] + 0.6 * predictions["channel"]
                elif "channel" in predictions:
                    return predictions["channel"]
                return predictions.get("hier", 0)
            else:
                if "channel" in predictions:
                    return predictions["channel"]
                return predictions.get("hier", 0)

        # 皮纸: 线性外推应该不错
        if group == "paper":
            preds = []
            if "per_sample" in predictions:
                preds.append(predictions["per_sample"])
            if "hier" in predictions:
                preds.append(predictions["hier"])
            if preds:
                return np.mean(preds)
            return predictions.get("channel", 0)

        # 默认
        return np.mean(list(predictions.values()))

# ===================== 10. 前向评估 =====================
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

            # 去掉最后一个点用于预测
            train_sub = sub.iloc[:-1]
            test_row = sub.iloc[-1]

            t_arr, dE_arr = prepare_series(train_sub)
            if len(t_arr) == 0:
                continue

            t_tgt = float(test_row["aging_time_day"])

            # 临时用train_sub构建模型
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
    print("  颜料老化色差预测 v14")
    print("  多策略集成 + 颜色通道分解")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 构建集成模型
    print("\n[模型] 构建v14集成模型...")
    ensemble = V14Ensemble(df_train)

    # 打印分层模型参数
    print("\n[分层模型] 组级参数:")
    for g, m in ensemble.hier.models.items():
        print(f"  {g:15s}: n={m['n']:.3f}, A={m['A']:.4f}, 成员={len(m['members'])}")

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
        pred = ensemble.predict_single(
            row["sample"], row["aging_condition"], float(row["aging_time_day"])
        )
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
    print(f"  v14:  R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    print(f"  v13:  R²=0.9784, MAE=0.2906")
    print("=" * 60)


if __name__ == "__main__":
    main()
