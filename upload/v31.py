"""
颜料老化色差预测 - v31
基于v30b + 曙红专用改进

改进：
1. 曙红组：使用比例外推法 (ratio-based extrapolation)
   - 组级增长比例中位数 ≈ 1.5 (每6天dE增长50%)
   - 对个体用个体t18_dE * 组ratio^((t_pred-18)/6) 外推
   - 曙红7特殊处理（dE下降，可能是测量误差）
2. 其他组保持v30b的最优权重
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = SCRIPT_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV  = SCRIPT_DIR / "baseline_and_data" / "paint_aging_testset.csv"
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
        from scipy.optimize import curve_fit
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

def predict_model(model, t_pred):
    if model is None: return None
    if model["type"] == "power": return max(model["A"] * (t_pred ** model["n"]), 0)
    elif model["type"] == "linear": return max(model["a"] + model["b"] * t_pred, 0)
    elif model["type"] == "log": return model["A"] * np.log(1 + model["k"] * t_pred)
    elif model["type"] == "mm": return max(model["A"] * t_pred / (model["B"] + t_pred), 0)
    elif model["type"] == "sqrt": return max(model["A"] * np.sqrt(t_pred) + model["B"], 0)
    return None

def best_model(t, y, min_score=-1.0):
    models = [fit_power_law(t, y), fit_log(t, y), fit_linear(t, y), fit_mm(t, y), fit_sqrt(t, y)]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models: return None
    return max(models, key=lambda m: m["score"])


class RobustGroupModel:
    def __init__(self, df_train):
        self.group_models = {}
        self.group_medians = {}
        self.group_channel_medians = {}
        self.shu_red_ratios = []  # 曙红组级增长比例
        self.shu_red_channel_rates = {}  # 曙红通道速率
        self._build(df_train)

    def _build(self, df_train):
        # 曙红组分析
        shu_red_samples = [s for s in df_train["sample"].unique() if detect_group(s) == "shu_red"]
        for sample in shu_red_samples:
            sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "UV")]
            sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
            if len(sub) >= 2:
                dE_12 = sub["dietaE"].values[0]
                dE_18 = sub["dietaE"].values[-1]
                if dE_12 > 0.01:
                    self.shu_red_ratios.append(dE_18 / dE_12)

        # 曙红通道速率
        dL_rates, da_rates, db_rates = [], [], []
        for sample in shu_red_samples:
            sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "UV")]
            sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
            if len(sub) >= 2:
                t = sub["aging_time_day"].values
                L0 = sub["L0"].iloc[0]; a0 = sub["a0"].iloc[0]; b0 = sub["b0"].iloc[0]
                L = sub["L"].values; a = sub["a"].values; b = sub["b"].values
                dt = t[-1] - t[-2]
                if dt > 0:
                    dL_rates.append((L[-1] - L0 - (L[-2] - L0)) / dt)
                    da_rates.append((a[-1] - a0 - (a[-2] - a0)) / dt)
                    db_rates.append((b[-1] - b0 - (b[-2] - b0)) / dt)

        self.shu_red_channel_rates = {
            "dL": np.median(dL_rates) if dL_rates else -0.05,
            "da": np.median(da_rates) if da_rates else -0.10,
            "db": np.median(db_rates) if db_rates else -0.01,
        }

        # 曙红组级通道中位数
        ch_12, ch_18 = {"dL": [], "da": [], "db": []}, {"dL": [], "da": [], "db": []}
        for sample in shu_red_samples:
            sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == "UV")]
            sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
            for _, row in sub.iterrows():
                t = int(row["aging_time_day"])
                dL = row["L"] - row["L0"]
                da = row["a"] - row["a0"]
                db = row["b"] - row["b0"]
                if t == 12:
                    ch_12["dL"].append(dL); ch_12["da"].append(da); ch_12["db"].append(db)
                elif t == 18:
                    ch_18["dL"].append(dL); ch_18["da"].append(da); ch_18["db"].append(db)

        self.shu_red_ch_medians = {
            12: {k: np.median(v) for k, v in ch_12.items()},
            18: {k: np.median(v) for k, v in ch_18.items()},
        }

        # 其他组
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

    def predict_group_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key in self.group_models: return predict_model(self.group_models[key], t_pred)
        return None

    def predict_group_channel_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key not in self.group_channel_medians: return None
        times, ch_meds = self.group_channel_medians[key]
        if not times: return None
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

    def predict_shu_red_ratio(self, sample, t_pred):
        """曙红专用：比例外推"""
        # 组级比例
        if not self.shu_red_ratios:
            return None
        # 去掉异常值(ratio < 0.8)
        clean_ratios = [r for r in self.shu_red_ratios if r > 0.8]
        if not clean_ratios:
            clean_ratios = self.shu_red_ratios
        ratio_per_6days = np.median(clean_ratios)  # ~1.5

        return ratio_per_6days

    def predict_shu_red_channel(self, sample, t_pred, dE_at_18):
        """曙红通道级外推：用组级通道速率"""
        ch18 = self.shu_red_ch_medians.get(18, {"dL": -0.57, "da": -1.57, "db": -0.99})
        ch12 = self.shu_red_ch_medians.get(12, {"dL": -0.22, "da": -0.73, "db": -0.92})

        # 组级通道速率 (per day)
        dL_rate = (ch18["dL"] - ch12["dL"]) / 6
        da_rate = (ch18["da"] - ch12["da"]) / 6
        db_rate = (ch18["db"] - ch12["db"]) / 6

        # 个体在t=18的通道值比例
        # 从dE反推：个体dE_18和组dE_18的比例作为缩放因子
        grp_dE_18 = np.sqrt(ch18["dL"]**2 + ch18["da"]**2 + ch18["db"]**2)
        if grp_dE_18 > 0.01 and dE_at_18 > 0.01:
            scale = dE_at_18 / grp_dE_18
        else:
            scale = 1.0

        # 组级通道外推到t_pred
        dt = t_pred - 18
        dL_ext = ch18["dL"] + dL_rate * dt
        da_ext = ch18["da"] + da_rate * dt
        db_ext = ch18["db"] + db_rate * dt

        # 缩放
        dE_ext = np.sqrt((dL_ext*scale)**2 + (da_ext*scale)**2 + (db_ext*scale)**2)
        return float(max(dE_ext, 0))


class V31Predictor:
    def __init__(self, df_train):
        self.df_train = df_train
        self.rgm = RobustGroupModel(df_train)
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
                self.sample_models[key] = {"model": model, "tc": tc, "dEc": dEc}

    def predict(self, sample, cond, t_pred):
        group = detect_group(sample)
        sub = self.df_train[(self.df_train["sample"] == sample) & (self.df_train["aging_condition"] == cond)].sort_values("aging_time_day")
        if len(sub) == 0: return float(self.df_train[TARGET].mean())
        t_arr, dE_arr = prepare_series(sub)
        if len(t_arr) == 0: return 0.0
        tc, dEc = remove_outliers(t_arr, dE_arr)
        key = f"{sample}_{cond}"

        # 曙红组专用策略
        if group == "shu_red":
            return self._predict_shu_red(sample, tc, dEc, t_pred)

        # 其他组使用v30b最优权重
        v30b_weights = {
            "dye": {"ind": 0.655, "grp": 0.0, "ch": 0.042, "lin": 0.296, "scaled": 0.007},
            "paper": {"ind": 0.008, "grp": 0.0, "ch": 0.0, "lin": 0.793, "scaled": 0.199},
            "jade_green": {"ind": 0.0, "grp": 0.261, "ch": 0.561, "lin": 0.0, "scaled": 0.179},
            "cobalt_blue": {"ind": 0.0, "grp": 0.691, "ch": 0.0, "lin": 0.0, "scaled": 0.309},
            "other": {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2},
        }

        w = v30b_weights.get(group, v30b_weights["other"])
        p_ind = predict_model(self.sample_models[key]["model"], t_pred) if key in self.sample_models else None
        p_grp = self.rgm.predict_group_dE(group, cond, t_pred)
        p_ch = self.rgm.predict_group_channel_dE(group, cond, t_pred)
        p_lin = None
        if len(tc) >= 2:
            rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
            p_lin = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)
        p_scaled = None
        if p_grp is not None and len(tc) >= 1:
            gmk = f"{group}_{cond}"
            if gmk in self.rgm.group_medians:
                gm = self.rgm.group_medians[gmk]
                gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
                if gv > 0.01: p_scaled = p_grp * dEc[-1] / gv

        strategies = {"ind": p_ind, "grp": p_grp, "ch": p_ch, "lin": p_lin, "scaled": p_scaled}
        ws, wp = 0, 0
        for sn, sv in strategies.items():
            if sv is not None and w.get(sn, 0) > 0:
                ws += w[sn]
                wp += w[sn] * sv
        return wp / ws if ws > 0 else (np.mean([v for v in strategies.values() if v is not None]) or 0.0)

    def _predict_shu_red(self, sample, tc, dEc, t_pred):
        """曙红专用预测"""
        # 策略A: 个体速率线性外推
        p_lin = None
        if len(tc) >= 2:
            rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
            p_lin = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)

        # 策略B: 组级比例外推
        ratio = self.rgm.predict_shu_red_ratio(sample, t_pred)
        p_ratio = None
        if ratio and len(tc) >= 1:
            n_periods = (t_pred - tc[-1]) / 6  # 每6天一个周期
            p_ratio = dEc[-1] * (ratio ** n_periods)

        # 策略C: 缩放组通道
        p_ch_scaled = self.rgm.predict_shu_red_channel(sample, t_pred, dEc[-1]) if len(tc) >= 1 else None

        # 策略D: 组级dE模型
        p_grp = self.rgm.predict_group_dE("shu_red", "UV", t_pred)
        p_grp_scaled = None
        if p_grp and len(tc) >= 1:
            gmk = "shu_red_UV"
            if gmk in self.rgm.group_medians:
                gm = self.rgm.group_medians[gmk]
                gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
                if gv > 0.01: p_grp_scaled = p_grp * dEc[-1] / gv

        # 策略E: 组通道分解
        p_ch = self.rgm.predict_group_channel_dE("shu_red", "UV", t_pred)

        # 策略F: 个体模型
        key = f"{sample}_UV"
        p_ind = predict_model(self.sample_models[key]["model"], t_pred) if key in self.sample_models else None

        # 检测异常个体（曙红7，dE下降）
        is_declining = len(tc) >= 2 and dEc[-1] < dEc[-2]

        if is_declining:
            # 异常个体：更多依赖组级
            candidates = []
            weights = []
            if p_grp_scaled is not None:
                candidates.append(p_grp_scaled); weights.append(0.5)
            if p_ratio is not None:
                candidates.append(p_ratio); weights.append(0.3)
            if p_ch is not None:
                candidates.append(p_ch); weights.append(0.2)
            if not candidates: return p_lin if p_lin is not None else dEc[-1]
            ws = sum(weights)
            return sum(c * w for c, w in zip(candidates, weights)) / ws

        # 正常个体：加权融合
        # 曙红组LOLO搜索最优权重
        candidates = []
        weights = []

        if p_lin is not None:
            candidates.append(p_lin); weights.append(0.25)
        if p_ratio is not None:
            candidates.append(p_ratio); weights.append(0.25)
        if p_ch_scaled is not None:
            candidates.append(p_ch_scaled); weights.append(0.20)
        if p_grp_scaled is not None:
            candidates.append(p_grp_scaled); weights.append(0.15)
        if p_ch is not None:
            candidates.append(p_ch); weights.append(0.10)
        if p_ind is not None:
            candidates.append(p_ind); weights.append(0.05)

        if not candidates:
            return p_lin if p_lin is not None else dEc[-1]

        ws = sum(weights)
        return sum(c * w for c, w in zip(candidates, weights)) / ws


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
        "n": len(y_true), "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "details": details,
    }


def main():
    print("=" * 60)
    print("  颜料老化色差预测 v31")
    print("  v30b基础 + 曙红专用改进")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    print("\n[评估] 真LOLO评估...")
    metrics = true_lolo_eval(df_train)
    print(f"  n={metrics['n']}, R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    print("\n[评估] 按组:")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        gd = [d for d in metrics["details"] if d["group"] == g]
        if len(gd) >= 2:
            yt = np.array([d["y_true"] for d in gd])
            yp = np.array([d["y_pred"] for d in gd])
            print(f"  {g:15s}: R²={r2_score(yt, yp):+.4f}, MAE={mean_absolute_error(yt, yp):.4f}, n={len(gd)}")

    print("\n[LOLO详情] 曙红组:")
    for d in metrics["details"]:
        if d["group"] == "shu_red":
            print(f"  {d['sample']:20s} t={d['t']:5.0f}  true={d['y_true']:.4f}  pred={d['y_pred']:.4f}  err={abs(d['y_true']-d['y_pred']):.4f}")

    # 测试集加权MAE (不含other)
    test_counts = {"dye": 5, "paper": 4, "shu_red": 14, "jade_green": 7, "cobalt_blue": 7}
    total_test = sum(test_counts.values())
    group_maes = {}
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
        gd = [d for d in metrics["details"] if d["group"] == g]
        if len(gd) >= 2:
            group_maes[g] = mean_absolute_error([d["y_true"] for d in gd], [d["y_pred"] for d in gd])
    weighted_mae = sum(group_maes.get(g, 0) * test_counts[g] for g in test_counts) / total_test
    print(f"\n[测试集加权MAE] = {weighted_mae:.4f}")

    print("\n[预测] 测试集...")
    predictor = V31Predictor(df_train)
    test_preds = []
    for _, row in df_test.iterrows():
        pred = predictor.predict(row["sample"], row["aging_condition"], float(row["aging_time_day"]))
        test_preds.append(float(max(pred, 0)))

    test_preds = np.array(test_preds)
    print(f"  范围: [{test_preds.min():.4f}, {test_preds.max():.4f}], 均值: {test_preds.mean():.4f}")

    out_csv = SCRIPT_DIR / "baseline_and_data" / "predict_out.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(out_csv, index=False)
    print(f"\n[完成] {out_csv}")

    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d) → {test_preds[i]:.4f}")

    # 保存
    download_dir = Path("/home/z/my-project/download")
    pd.DataFrame({TARGET: test_preds}).to_csv(download_dir / "predict_out_v31.csv", index=False)

    # 生成ensemble
    v14 = pd.read_csv(SCRIPT_DIR / "predict_out_v14.csv")[TARGET].values
    v30 = pd.read_csv(SCRIPT_DIR / "predict_out_v30.csv")[TARGET].values
    for w31, label in [(0.3, "0.3v14_0.7v31"), (0.5, "0.5v14_0.5v31"), (0.7, "0.7v14_0.3v31")]:
        ens = w31 * v14 + (1 - w31) * test_preds
        pd.DataFrame({TARGET: ens}).to_csv(download_dir / f"predict_out_v14_v31_{label}.csv", index=False)
    for w31, label in [(0.3, "0.3v30_0.7v31"), (0.5, "0.5v30_0.5v31")]:
        ens = w31 * v30 + (1 - w31) * test_preds
        pd.DataFrame({TARGET: ens}).to_csv(download_dir / f"predict_out_v30_v31_{label}.csv", index=False)

    print(f"\nv31 测试集加权MAE = {weighted_mae:.4f}")


if __name__ == "__main__":
    main()
