"""
颜料老化色差预测 - v30b
LOLO引导权重优化 (预计算策略预测值，然后搜索权重)

核心思想：
1. 对每个LOLO评估点，预计算所有5种策略的预测值
2. 然后在缓存的预测值上做权重搜索 (极快)
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


# ===================== 预计算所有LOLO策略预测值 =====================
def precompute_lolo_strategies(df_train):
    """对每个LOLO评估点，计算所有5种策略的预测值"""
    records = []
    strategy_names = ["ind", "grp", "ch", "lin", "scaled"]

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

            group = detect_group(sample)
            t_pred = float(last_t)

            # 构建模型
            rgm = RobustGroupModel(train_df)

            # 获取训练序列
            t_sub = train_df[(train_df["sample"] == sample) & (train_df["aging_condition"] == cond)]
            t_arr, dE_arr = prepare_series(t_sub)
            if len(t_arr) == 0: continue
            tc, dEc = remove_outliers(t_arr, dE_arr)

            # 策略1: 个体模型
            p_ind = None
            if len(tc) >= 2:
                m = best_model(tc, dEc)
                p_ind = predict_model(m, t_pred)

            # 策略2: 组级dE
            p_grp = rgm.predict_group_dE(group, cond, t_pred)

            # 策略3: 组级通道
            p_ch = rgm.predict_group_channel_dE(group, cond, t_pred)

            # 策略4: 线性外推
            p_lin = None
            if len(tc) >= 2:
                rate = (dEc[-1] - dEc[-2]) / (tc[-1] - tc[-2]) if tc[-1] > tc[-2] else 0
                p_lin = max(dEc[-1] + rate * (t_pred - tc[-1]), 0)

            # 策略5: 缩放组模型
            p_scaled = None
            if p_grp is not None and len(tc) >= 1:
                gmk = f"{group}_{cond}"
                if gmk in rgm.group_medians:
                    gm = rgm.group_medians[gmk]
                    gv = gm["medians"][np.argmin(np.abs(gm["times"] - tc[-1]))]
                    if gv > 0.01:
                        p_scaled = p_grp * dEc[-1] / gv

            y_true = float(sub[sub["aging_time_day"] == last_t].iloc[-1][TARGET])

            record = {
                "sample": sample, "group": group, "cond": cond, "t": t_pred,
                "y_true": y_true,
                "ind": p_ind, "grp": p_grp, "ch": p_ch, "lin": p_lin, "scaled": p_scaled,
            }
            records.append(record)

    return pd.DataFrame(records)


def search_weights(lolo_df):
    """在预计算的LOLO结果上搜索最优权重"""
    strategy_names = ["ind", "grp", "ch", "lin", "scaled"]
    groups = ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]

    best_overall_weights = {}
    best_overall_mae = float("inf")

    # 对每个组搜索最优权重
    group_results = {}

    for group in groups:
        gdf = lolo_df[lolo_df["group"] == group]
        if len(gdf) < 2:
            group_results[group] = {"best_mae": float("inf"), "best_w": None}
            continue

        best_mae = float("inf")
        best_w = None

        np.random.seed(42)
        # 精细网格搜索
        for trial in range(5000):
            raw_w = np.random.dirichlet(np.ones(5) * 0.5)  # 更集中于某些权重
            # 允许一些权重为0
            mask = np.random.random(5) > 0.25
            raw_w *= mask
            if raw_w.sum() < 0.01: continue
            raw_w /= raw_w.sum()

            # 计算加权预测
            pred = np.zeros(len(gdf))
            for i, sn in enumerate(strategy_names):
                vals = gdf[sn].values.astype(float)
                w = raw_w[i]
                # 缺失值用其他策略平均替代
                valid = ~np.isnan(vals)
                if valid.sum() == 0:
                    continue
                pred[valid] += w * vals[valid]
                # 对缺失值，按该策略在其他点的平均比例填充
                if (~valid).any():
                    overall_mean = np.nanmean(vals)
                    if not np.isnan(overall_mean):
                        # 用有效预测的平均比例来填充
                        valid_preds = pred[valid]
                        valid_sum = np.sum([raw_w[j] * (~np.isnan(gdf[sn2].values.astype(float))).all() for j, sn2 in enumerate(strategy_names)])
                        if valid_sum > 0:
                            pred[~valid] += w * overall_mean * np.mean(valid_preds[valid] > 0) if (valid_preds > 0).any() else 0

            # 归一化（处理缺失）
            total_w = np.array([raw_w[i] if not np.isnan(gdf[sn].values[j]) else 0 
                               for j in range(len(gdf)) for i, sn in enumerate(strategy_names)])
            # 简化：只对有效值计算
            mae = 0
            count = 0
            for idx in range(len(gdf)):
                row = gdf.iloc[idx]
                ws = 0
                wp = 0
                for i, sn in enumerate(strategy_names):
                    v = row[sn]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        ws += raw_w[i]
                        wp += raw_w[i] * v
                if ws > 0:
                    mae += abs(row["y_true"] - wp / ws)
                    count += 1

            if count > 0:
                mae /= count
                if mae < best_mae:
                    best_mae = mae
                    best_w = raw_w.copy()

        group_results[group] = {"best_mae": best_mae, "best_w": best_w}
        print(f"  {group:15s}: best MAE = {best_mae:.4f}, weights = {dict(zip(strategy_names, np.round(best_w, 3)))}")

    return group_results


def main():
    print("=" * 60)
    print("  颜料老化色差预测 v30b")
    print("  预计算策略 + 权重搜索")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 预计算所有LOLO策略预测值
    print("\n[预计算] 所有LOLO策略预测值...")
    lolo_df = precompute_lolo_strategies(df_train)
    print(f"  共 {len(lolo_df)} 个LOLO评估点")

    # 基线：v28默认权重
    v28_weights = {
        "cobalt_blue": {"ind": 0.0, "grp": 0.3, "ch": 0.5, "lin": 0.0, "scaled": 0.2},
        "jade_green": {"ind": 0.3, "grp": 0.2, "ch": 0.3, "lin": 0.0, "scaled": 0.2},
        "shu_red": {"ind": 0.2, "grp": 0.3, "ch": 0.0, "lin": 0.0, "scaled": 0.5},
        "dye": {"ind": 0.8, "grp": 0.2, "ch": 0.0, "lin": 0.0, "scaled": 0.0},
        "paper": {"ind": 0.3, "grp": 0.2, "ch": 0.0, "lin": 0.3, "scaled": 0.2},
    }
    strategy_names = ["ind", "grp", "ch", "lin", "scaled"]

    # 评估v28基线
    print("\n[基线] v28默认权重评估:")
    for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
        gdf = lolo_df[lolo_df["group"] == group]
        w = v28_weights.get(group, {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2})
        errors = []
        for idx in range(len(gdf)):
            row = gdf.iloc[idx]
            ws, wp = 0, 0
            for i, sn in enumerate(strategy_names):
                v = row[sn]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    ws += w[sn]
                    wp += w[sn] * v
            if ws > 0: errors.append(abs(row["y_true"] - wp / ws))
        mae = np.mean(errors) if errors else 0
        print(f"  {group:15s}: MAE = {mae:.4f}")

    # 搜索最优权重
    print("\n[搜索] 对每组搜索最优权重...")
    group_results = search_weights(lolo_df)

    # 组合最优权重
    final_weights = {}
    for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
        r = group_results[group]
        if r["best_w"] is not None:
            final_weights[group] = dict(zip(strategy_names, r["best_w"]))
        else:
            final_weights[group] = v28_weights.get(group, {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2})
    final_weights["other"] = {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2}

    # 用最优权重重新评估
    print("\n[最终] 最优权重评估:")
    all_errors = []
    for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        gdf = lolo_df[lolo_df["group"] == group]
        w = final_weights.get(group, {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2})
        errors = []
        for idx in range(len(gdf)):
            row = gdf.iloc[idx]
            ws, wp = 0, 0
            for i, sn in enumerate(strategy_names):
                v = row[sn]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    ws += w.get(sn, 0)
                    wp += w.get(sn, 0) * v
            if ws > 0: errors.append(abs(row["y_true"] - wp / ws))
        mae = np.mean(errors) if errors else 0
        all_errors.extend(errors)
        if len(gdf) >= 2:
            print(f"  {group:15s}: MAE = {mae:.4f}")

    print(f"\n  总体 MAE = {np.mean(all_errors):.4f}")

    # 用最优权重生成测试集预测
    print("\n[预测] 测试集...")
    # 需要重新构建模型
    class FastPredictor:
        def __init__(self, df_train, weights):
            self.df_train = df_train
            self.weights = weights
            self.rgm = RobustGroupModel(df_train)
            self.sample_models = {}
            self._build()

        def _build(self):
            for sample in self.df_train["sample"].unique():
                for cond in self.df_train[self.df_train["sample"] == sample]["aging_condition"].unique():
                    sub = self.df_train[(self.df_train["sample"] == sample) & (self.df_train["aging_condition"] == cond)]
                    t, dE = prepare_series(sub)
                    if len(t) < 2: continue
                    tc, dEc = remove_outliers(t, dE)
                    if len(tc) < 2: continue
                    key = f"{sample}_{cond}"
                    self.sample_models[key] = {"model": best_model(tc, dEc), "tc": tc, "dEc": dEc}

        def predict(self, sample, cond, t_pred):
            group = detect_group(sample)
            w = self.weights.get(group, {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2})
            sub = self.df_train[(self.df_train["sample"] == sample) & (self.df_train["aging_condition"] == cond)].sort_values("aging_time_day")
            if len(sub) == 0: return float(self.df_train[TARGET].mean())
            t_arr, dE_arr = prepare_series(sub)
            if len(t_arr) == 0: return 0.0
            tc, dEc = remove_outliers(t_arr, dE_arr)
            key = f"{sample}_{cond}"

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

    predictor = FastPredictor(df_train, final_weights)
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
    pd.DataFrame({TARGET: test_preds}).to_csv(download_dir / "predict_out_v30.csv", index=False)

    # 也生成v14+v30 ensemble
    v14 = pd.read_csv(SCRIPT_DIR / "predict_out_v14.csv")[TARGET].values
    v28 = pd.read_csv(SCRIPT_DIR / "predict_out_v28.csv")[TARGET].values
    for w14, label in [(0.3, "0.3v14_0.7v30"), (0.5, "0.5v14_0.5v30"), (0.7, "0.7v14_0.3v30")]:
        ens = w14 * v14 + (1 - w14) * test_preds
        pd.DataFrame({TARGET: ens}).to_csv(download_dir / f"predict_out_v14_v30_{label}.csv", index=False)
    print(f"\n[生成] v14+v30 ensemble文件")
    print(f"[最优权重]")
    for g, w in final_weights.items():
        print(f"  {g:15s}: {w}")


if __name__ == "__main__":
    main()
