"""
颜料老化色差预测 - v30
基于v28的LOLO引导权重优化

策略：对每个组，搜索策略权重的最优组合
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from itertools import product as iter_product

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

def fit_linear(t, y):
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

def fit_log(t, y):
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

def fit_mm(t, y):
    mask = (t > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    t, y = t[mask], y[mask]
    try:
        from scipy.optimize import curve_fit
        t_max = t.max()
        y_max = y.max()
        t_norm = t / t_max
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

def fit_sqrt(t, y):
    mask = (t > 0)
    if mask.sum() < 2:
        return None
    t, y = t[mask], y[mask]
    A_mat = np.vstack([np.sqrt(t), np.ones_like(t)]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        pred = coeffs[0] * np.sqrt(t) + coeffs[1]
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "sqrt", "A": coeffs[0], "B": coeffs[1], "score": score}
    except:
        return None

def predict_model(model, t_pred):
    if model is None:
        return None
    if model["type"] == "power":
        return max(model["A"] * (t_pred ** model["n"]), 0)
    elif model["type"] == "linear":
        return max(model["a"] + model["b"] * t_pred, 0)
    elif model["type"] == "log":
        return model["A"] * np.log(1 + model["k"] * t_pred)
    elif model["type"] == "mm":
        return max(model["A"] * t_pred / (model["B"] + t_pred), 0)
    elif model["type"] == "sqrt":
        return max(model["A"] * np.sqrt(t_pred) + model["B"], 0)
    return None

def best_model(t, y, min_score=-1.0):
    models = [fit_power_law(t, y), fit_log(t, y), fit_linear(t, y), fit_mm(t, y), fit_sqrt(t, y)]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models:
        return None
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
                time_data = {}
                channel_data = {}
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
                if len(times) < 2:
                    continue

                medians_t = np.array(times, dtype=float)
                medians_y = np.array([np.median(time_data[t]) for t in times])

                key = f"{group}_{cond}"
                self.group_medians[key] = {"times": medians_t, "medians": medians_y}
                model = best_model(medians_t, medians_y)
                if model:
                    self.group_models[key] = model

                ch_times = sorted(channel_data.keys())
                ch_meds = {t: {
                    "dL": np.median(channel_data[t]["dL"]),
                    "da": np.median(channel_data[t]["da"]),
                    "db": np.median(channel_data[t]["db"]),
                } for t in ch_times}
                self.group_channel_medians[key] = ch_times, ch_meds

    def predict_group_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key in self.group_models:
            return predict_model(self.group_models[key], t_pred)
        return None

    def predict_group_channel_dE(self, group, cond, t_pred):
        key = f"{group}_{cond}"
        if key not in self.group_channel_medians:
            return None
        times, ch_meds = self.group_channel_medians[key]
        if not times:
            return None
        if t_pred <= times[0]:
            ch = ch_meds[times[0]]
        elif t_pred >= times[-1]:
            ch = ch_meds[times[-1]]
        else:
            for i in range(len(times) - 1):
                if times[i] <= t_pred <= times[i+1]:
                    frac = (t_pred - times[i]) / (times[i+1] - times[i])
                    ch = {}
                    for k in ["dL", "da", "db"]:
                        ch[k] = ch_meds[times[i]][k] * (1 - frac) + ch_meds[times[i+1]][k] * frac
                    break
            else:
                ch = ch_meds[times[-1]]
        dE = np.sqrt(ch["dL"]**2 + ch["da"]**2 + ch["db"]**2)
        return float(max(dE, 0))


class FlexiblePredictor:
    """支持参数化权重的预测器"""

    def __init__(self, df_train, group_weights=None):
        self.df_train = df_train
        self.robust_model = RobustGroupModel(df_train)
        self.sample_models = {}
        self._build_sample_models(df_train)
        # group_weights: {group: {strategy_name: weight}}
        self.group_weights = group_weights or self._default_weights()

    def _default_weights(self):
        return {
            "cobalt_blue": {"ind": 0.0, "grp": 0.3, "ch": 0.5, "lin": 0.0, "scaled": 0.2},
            "jade_green": {"ind": 0.3, "grp": 0.2, "ch": 0.3, "lin": 0.0, "scaled": 0.2},
            "shu_red": {"ind": 0.2, "grp": 0.3, "ch": 0.0, "lin": 0.0, "scaled": 0.5},
            "dye": {"ind": 0.8, "grp": 0.2, "ch": 0.0, "lin": 0.0, "scaled": 0.0},
            "paper": {"ind": 0.3, "grp": 0.2, "ch": 0.0, "lin": 0.3, "scaled": 0.2},
            "other": {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2},
        }

    def _build_sample_models(self, df_train):
        for sample in df_train["sample"].unique():
            for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
                sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
                t, dE = prepare_series(sub)
                if len(t) < 2:
                    continue
                tc, dEc = remove_outliers(t, dE)
                if len(tc) < 2:
                    continue
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

        p_scaled_group = None
        if p_group is not None and len(tc) >= 1:
            group_median_key = f"{group}_{cond}"
            if group_median_key in self.robust_model.group_medians:
                gmedians = self.robust_model.group_medians[group_median_key]
                group_val_at_t = gmedians["medians"][np.argmin(np.abs(gmedians["times"] - tc[-1]))]
                if group_val_at_t > 0.01:
                    scale = dEc[-1] / group_val_at_t
                    p_scaled_group = p_group * scale

        # 用权重混合
        w = self.group_weights.get(group, self._default_weights()[group])

        strategies = {
            "ind": p_individual,
            "grp": p_group,
            "ch": p_channel,
            "lin": p_linear,
            "scaled": p_scaled_group,
        }

        total_w = 0
        weighted_sum = 0
        for sname, sw in w.items():
            if strategies.get(sname) is not None and sw > 0:
                weighted_sum += sw * strategies[sname]
                total_w += sw

        if total_w > 0:
            return weighted_sum / total_w

        # fallback
        candidates = [p for p in strategies.values() if p is not None]
        return np.mean(candidates) if candidates else 0.0


def true_lolo_eval_weighted(df_train, group_weights):
    """用指定权重进行LOLO评估"""
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
            last_t = sub["aging_time_day"].max()
            train_df = df_train[~(
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond) &
                (df_train["aging_time_day"] == last_t)
            )]
            predictor = FlexiblePredictor(train_df, group_weights)
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


def search_best_weights(df_train):
    """对每个组搜索最优策略权重"""
    print("\n[搜索] 对每个组搜索最优权重...")
    strategy_names = ["ind", "grp", "ch", "lin", "scaled"]
    best_overall = None
    best_mae = float("inf")

    # 只搜索测试集中出现的组
    groups = ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]

    # 对每个组独立搜索
    group_best_weights = {}
    group_best_mae = {}

    for group in groups:
        best_gw = None
        best_gw_mae = float("inf")

        # 网格搜索：权重步长0.1
        # 5个策略，每个0-1，归一化
        # 采样搜索 (太多组合了)
        np.random.seed(42)
        n_trials = 500

        for trial in range(n_trials):
            # 生成随机权重
            raw_w = np.random.dirichlet(np.ones(5))
            # 让某些策略可以为0
            mask = np.random.random(5) > 0.3
            raw_w *= mask
            if raw_w.sum() < 1e-6:
                continue
            raw_w /= raw_w.sum()

            w = {s: raw_w[i] for i, s in enumerate(strategy_names)}

            # 先用其他组的默认权重，这个组用搜索的权重
            default_w = {
                "cobalt_blue": {"ind": 0.0, "grp": 0.3, "ch": 0.5, "lin": 0.0, "scaled": 0.2},
                "jade_green": {"ind": 0.3, "grp": 0.2, "ch": 0.3, "lin": 0.0, "scaled": 0.2},
                "shu_red": {"ind": 0.2, "grp": 0.3, "ch": 0.0, "lin": 0.0, "scaled": 0.5},
                "dye": {"ind": 0.8, "grp": 0.2, "ch": 0.0, "lin": 0.0, "scaled": 0.0},
                "paper": {"ind": 0.3, "grp": 0.2, "ch": 0.0, "lin": 0.3, "scaled": 0.2},
                "other": {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2},
            }
            default_w[group] = w

            metrics = true_lolo_eval_weighted(df_train, default_w)

            # 计算该组的MAE
            gd = [d for d in metrics["details"] if d["group"] == group]
            if len(gd) >= 2:
                gmae = mean_absolute_error(
                    [d["y_true"] for d in gd],
                    [d["y_pred"] for d in gd]
                )
                if gmae < best_gw_mae:
                    best_gw_mae = gmae
                    best_gw = w.copy()

        group_best_weights[group] = best_gw
        group_best_mae[group] = best_gw_mae
        print(f"  {group:15s}: best MAE = {best_gw_mae:.4f}, weights = {best_gw}")

    return group_best_weights, group_best_mae


def main():
    print("=" * 60)
    print("  颜料老化色差预测 v30")
    print("  LOLO引导的组级权重优化")
    print("=" * 60)
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 搜索最优权重
    best_weights, best_maes = search_best_weights(df_train)

    # 组合所有组的最佳权重
    final_weights = {
        "cobalt_blue": best_weights["cobalt_blue"],
        "jade_green": best_weights["jade_green"],
        "shu_red": best_weights["shu_red"],
        "dye": best_weights["dye"],
        "paper": best_weights["paper"],
        "other": {"ind": 0.2, "grp": 0.2, "ch": 0.2, "lin": 0.2, "scaled": 0.2},
    }

    # 评估最终模型
    print("\n[评估] 最终模型LOLO评估...")
    metrics = true_lolo_eval_weighted(df_train, final_weights)
    print(f"  n={metrics['n']}, R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    print("\n[评估] 按组:")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        gd = [d for d in metrics["details"] if d["group"] == g]
        if len(gd) >= 2:
            yt = np.array([d["y_true"] for d in gd])
            yp = np.array([d["y_pred"] for d in gd])
            print(f"  {g:15s}: R²={r2_score(yt, yp):+.4f}, MAE={mean_absolute_error(yt, yp):.4f}, n={len(gd)}")

    # 预测测试集
    print("\n[预测] 测试集...")
    predictor = FlexiblePredictor(df_train, final_weights)
    test_preds = []
    for _, row in df_test.iterrows():
        pred = predictor.predict(row["sample"], row["aging_condition"], float(row["aging_time_day"]))
        test_preds.append(float(max(pred, 0)))

    test_preds = np.array(test_preds)
    print(f"  范围: [{test_preds.min():.4f}, {test_preds.max():.4f}], 均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] {OUT_CSV}")

    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d) → {test_preds[i]:.4f}")

    # 保存
    download_dir = Path("/home/z/my-project/download")
    pd.DataFrame({TARGET: test_preds}).to_csv(download_dir / "predict_out_v30.csv", index=False)
    print(f"[保存] {download_dir / 'predict_out_v30.csv'}")

    # 打印最优权重
    print("\n[最优权重]")
    for g, w in final_weights.items():
        print(f"  {g:15s}: {w}")


if __name__ == "__main__":
    main()
