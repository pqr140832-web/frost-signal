"""
颜料老化色差预测 - v31
基于v28/v30的深度改进版

核心改进思路：
1. better extrapolation: 对皮纸组(t_train max=15, t_pred=40)做更好的饱和曲线建模
2. 曙红组: 用组内所有样本的时间序列做协同建模，而非单独外推
3. other组(中国画/颜彩/矿物颜料): 测试集t_pred=62恰好等于训练最大t，不需外推！直接用训练最后值趋势
4. 通道级建模改进: 对ΔL/Δa/Δb分别做group-level + individual scaling
5. 染料组: t=4→5的短距离外推，用最后两点的增量更准确
6. ensemble优化: 基于LOLO验证的加权策略
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/home/z/my-project/upload/baseline_and_data")
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV  = DATA_DIR / "paint_aging_testset.csv"
TARGET = "dietaE"

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
    except:
        return None

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
    except:
        return None

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
    except:
        return None

def fit_exp_decay(t, y):
    """y = A * (1 - exp(-k*t)): 快速饱和模型，适合皮纸等"""
    mask = (t > 0) & (y >= 0)
    if mask.sum() < 3: return None
    t, y = t[mask], y[mask]
    try:
        y_max = y.max() + 1e-6
        def expdec(tn, k):
            return 1 - np.exp(-k * tn)
        popt, _ = curve_fit(expdec, t / t.max(), y / y_max, p0=[1.0], bounds=(0.01, 10), maxfev=5000)
        k = popt[0]
        pred = y_max * expdec(t / t.max(), k)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "exp_decay", "A": y_max, "k": k / t.max(), "score": score}
    except:
        return None

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
        self._build(df_train)

    def _build(self, df_train):
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

                # 新增: 对组级通道数据也建模(用于外推)
                ch_models = {}
                for ch_name in ["dL", "da", "db"]:
                    ch_vals = np.array([ch_meds[t][ch_name] for t in ch_times])
                    # 对通道值用线性模型(允许负值)
                    lin = fit_linear(medians_t, ch_vals)
                    # 对绝对值用幂律
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

        # 优先用通道模型外推
        if key in self.group_channel_models and t_pred > times[-1]:
            ch_preds = {}
            for ch_name in ["dL", "da", "db"]:
                ch_model = self.group_channel_models[key][ch_name]
                # 策略1: 线性外推
                p_lin = predict_model(ch_model["linear"], t_pred)
                # 策略2: 绝对值模型+符号
                p_abs = None
                if ch_model["abs_model"]:
                    p_abs_val = predict_model(ch_model["abs_model"], t_pred)
                    if p_abs_val is not None:
                        p_abs = ch_model["sign"] * p_abs_val
                # 选择：线性优先(通道值通常变化比较线性)
                if p_lin is not None:
                    ch_preds[ch_name] = p_lin
                elif p_abs is not None:
                    ch_preds[ch_name] = p_abs
                else:
                    # fallback: 最后值
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


# ===================== 主预测器 =====================
class V31Predictor:
    def __init__(self, df_train):
        self.df_train = df_train
        self.robust_model = RobustGroupModel(df_train)
        self.sample_models = {}
        self._build_sample_models(df_train)

    def _build_sample_models(self, df_train):
        for sample in df_train["sample"].unique():
            for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
                sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
                t, dE = prepare_series(sub)
                if len(t) < 2: continue
                tc, dEc = remove_outliers(t, dE)
                if len(tc) < 2: continue
                model = best_model(tc, dEc)
                key = f"{sample}_{cond}"
                self.sample_models[key] = {
                    "model": model,
                    "t_last": tc[-1],
                    "dE_last": dEc[-1],
                    "t": tc,
                    "dE": dEc,
                }

    def predict(self, sample, cond, t_pred):
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

        key = f"{sample}_{cond}"
        tc, dEc = remove_outliers(t_arr, dE_arr)

        # 5种策略
        p_individual = None
        if key in self.sample_models:
            m = self.sample_models[key]["model"]
            p_individual = predict_model(m, t_pred)

        p_group = self.robust_model.predict_group_dE(group, cond, t_pred)
        p_channel = self.robust_model.predict_group_channel_dE(group, cond, t_pred)

        p_linear = None
        if len(tc) >= 2:
            rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
            p_linear = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)

        p_scaled = None
        if p_group is not None and len(tc) >= 1:
            gmk = f"{group}_{cond}"
            if gmk in self.robust_model.group_medians:
                gm = self.robust_model.group_medians[gmk]
                gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
                if gv > 0.01:
                    p_scaled = p_group * dEc[-1] / gv

        # 策略6: 基于个体通道缩放的组通道预测
        p_ch_scaled = None
        if p_channel is not None and len(tc) >= 1:
            gmk = f"{group}_{cond}"
            if gmk in self.robust_model.group_medians:
                gm = self.robust_model.group_medians[gmk]
                gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
                if gv > 0.01:
                    p_ch_scaled = p_channel * dEc[-1] / gv

        return self._select_strategy(group, tc, dEc, t_pred,
                                     p_individual, p_group, p_channel,
                                     p_linear, p_scaled, p_ch_scaled)

    def _select_strategy(self, group, t_train, dE_train, t_pred,
                         p_ind, p_grp, p_ch, p_lin, p_scaled, p_ch_scaled):
        candidates = {"ind": p_ind, "grp": p_grp, "ch": p_ch,
                      "lin": p_lin, "scaled": p_scaled, "ch_scaled": p_ch_scaled}
        valid = {k: v for k, v in candidates.items() if v is not None}

        if not valid:
            return 0.0

        t_max_train = t_train[-1] if len(t_train) > 0 else 0
        extrap_ratio = t_pred / t_max_train if t_max_train > 0 else 999

        if group == "cobalt_blue":
            # 钴蓝：数据噪声大，通道分解效果好
            # 测试t_pred=30, 训练到24, 外推1.25x
            if p_ch_scaled and p_grp:
                return 0.4 * p_ch_scaled + 0.3 * p_ch + 0.3 * p_grp
            elif p_ch and p_grp:
                return 0.5 * p_ch + 0.3 * p_grp + 0.2 * (p_scaled if p_scaled else p_grp)
            elif p_ch:
                return p_ch
            return p_grp if p_grp else np.mean(list(valid.values()))

        if group == "jade_green":
            # 翡翠绿：训练到24, 测试30, 外推1.25x
            if p_ch_scaled and p_ind:
                return 0.35 * p_ch_scaled + 0.25 * p_ch + 0.25 * p_ind + 0.15 * p_grp
            elif p_ind and p_grp:
                return 0.4 * p_ind + 0.3 * p_grp + 0.3 * (p_ch if p_ch else p_ind)
            return p_ind if p_ind else p_grp

        if group == "shu_red":
            # 曙红：训练到18, 测试24/30, 外推1.33~1.67x
            # 关键: 组内协同建模更重要
            if p_ch_scaled and p_ind:
                return 0.5 * p_ch_scaled + 0.3 * p_ind + 0.2 * p_grp
            elif p_scaled and p_ind:
                return 0.5 * p_scaled + 0.3 * p_ind + 0.2 * p_grp
            elif p_grp and p_ind:
                return 0.5 * p_grp + 0.5 * p_ind
            elif p_ind:
                return p_ind
            return p_grp if p_grp else np.mean(list(valid.values()))

        if group == "dye":
            # 染料组：训练到4, 测试5, 外推1.25x (距离短)
            # 短距离外推: 个体模型+线性外推最可靠
            if p_ind and p_lin:
                # 如果模型R2高，信任模型；否则偏线性
                score = self.sample_models.get(f"_{t_train[0]}", {}).get("model", {}).get("score", 0.5)
                if p_ind is not None and len(t_train) >= 4:
                    # 数据充足时信任个体模型
                    return 0.7 * p_ind + 0.3 * p_lin
                else:
                    return 0.5 * p_ind + 0.5 * p_lin
            elif p_ind:
                return p_ind
            elif p_lin:
                return p_lin
            return p_grp if p_grp else np.mean(list(valid.values()))

        if group == "paper":
            # 皮纸组：训练到15, 测试40, 外推2.67x (最大外推距离！)
            # 饱和曲线模型最适合大幅外推
            if p_grp and p_ch:
                # 组级饱和模型 + 通道分解
                return 0.4 * p_grp + 0.3 * p_ch + 0.3 * (p_ch_scaled if p_ch_scaled else p_grp)
            elif p_grp and p_lin:
                return 0.5 * p_grp + 0.3 * p_lin + 0.2 * (p_ind if p_ind else p_grp)
            elif p_grp:
                return p_grp
            return p_lin if p_lin else np.mean(list(valid.values()))

        # other组(中国画/颜彩/矿物颜料): 训练到62, 测试也是62
        # 几乎没有外推！直接用训练数据最后趋势
        if extrap_ratio <= 1.05:
            # 基本不外推：用最后已知值的趋势
            if p_ind and p_lin:
                return 0.5 * p_ind + 0.5 * p_lin
            elif p_ind:
                return p_ind
            elif p_lin:
                return p_lin
            return np.mean(list(valid.values()))
        else:
            # 有外推
            if p_ind and p_grp:
                return 0.5 * p_ind + 0.3 * p_grp + 0.2 * p_ch
            elif p_ind:
                return p_ind
            return np.mean(list(valid.values()))


# ===================== LOLO评估 =====================
def true_lolo_eval(df_train):
    y_true, y_pred = [], []
    details = []
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
            predictor = V31Predictor(train_df)
            test_row = sub[sub["aging_time_day"] == last_t].iloc[-1]
            pred = predictor.predict(sample, cond, float(test_row["aging_time_day"]))
            y_true.append(test_row[TARGET])
            y_pred.append(float(pred))
            details.append({
                "sample": sample, "group": detect_group(sample),
                "cond": cond, "t": float(test_row["aging_time_day"]),
                "y_true": test_row[TARGET], "y_pred": float(pred),
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


def main():
    print("=" * 60)
    print("  颜料老化色差预测 v31")
    print("  改进: 饱和曲线+通道外推+策略6")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    print("\n[评估] 真LOLO评估...")
    metrics = true_lolo_eval(df_train)
    print(f"  n={metrics['n']}, R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    print("\n[评估] 按组:")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        gd = [d for d in metrics["details"] if d["group"] == g]
        if len(gd) >= 2:
            yt = np.array([d["y_true"] for d in gd])
            yp = np.array([d["y_pred"] for d in gd])
            print(f"  {g:15s}: R2={r2_score(yt, yp):+.4f}, MAE={mean_absolute_error(yt, yp):.4f}, n={len(gd)}")

    print("\n[预测] 测试集...")
    predictor = V31Predictor(df_train)
    test_preds = []
    for _, row in df_test.iterrows():
        pred = predictor.predict(row["sample"], row["aging_condition"], float(row["aging_time_day"]))
        test_preds.append(float(max(pred, 0)))

    test_preds = np.array(test_preds)
    print(f"  范围: [{test_preds.min():.4f}, {test_preds.max():.4f}], 均值: {test_preds.mean():.4f}")

    # 保存
    out_csv = DATA_DIR / "predict_out.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(out_csv, index=False)
    print(f"\n[完成] {out_csv}")

    download_dir = Path("/home/z/my-project/download")
    pd.DataFrame({TARGET: test_preds}).to_csv(download_dir / "predict_out_v31.csv", index=False)

    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d) -> {test_preds[i]:.4f}")

    print(f"\nv31 LOLO: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")


if __name__ == "__main__":
    main()
