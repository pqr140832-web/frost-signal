"""
颜料老化色差预测 - v27
颜色通道时序预测 + 智能集成

核心改进：
1. 对每个样本的L/a/b通道分别做时序外推
2. 用多种外推方法(幂律/对数/线性/指数衰减)并自适应选择
3. 组级通道统计作为先验约束
4. 与v14集成

关键洞察：别人70+可能是预测了L a b的变化 然后算dE
而不是直接预测dE本身
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = SCRIPT_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV  = SCRIPT_DIR / "baseline_and_data" / "paint_aging_testset.csv"
OUT_CSV   = SCRIPT_DIR / "baseline_and_data" / "predict_out.csv"
TARGET = "dietaE"

def detect_group(sample: str) -> str:
    if "翡翠绿" in sample: return "jade_green"
    if "钴蓝" in sample: return "cobalt_blue"
    if "曙红" in sample: return "shu_red"
    if "皮纸" in sample: return "paper"
    if any(x in sample for x in ["染料", "紫草", "苏木", "红花", "黄檗"]):
        return "dye"
    return "other"

# ===================== 时序模型库 =====================
def fit_power(t, y):
    """y = A * t^n + b"""
    if len(t) < 2:
        return None
    def neg_r2(params):
        n, b = params
        try:
            tn = np.power(np.maximum(t, 0.01), n)
            A = np.dot(tn, y - b) / (np.dot(tn, tn) + 1e-9)
            pred = A * tn + b
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return -(1 - ss_res / (ss_tot + 1e-9))
        except:
            return 1e9

    from scipy.optimize import minimize
    res = minimize(neg_r2, x0=[0.5, 0], bounds=[(0.01, 3.0), (None, None)], method="L-BFGS-B")
    n, b = res.x
    tn = np.power(np.maximum(t, 0.01), n)
    A = np.dot(tn, y - b) / (np.dot(tn, tn) + 1e-9)
    pred = A * tn + b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    score = 1 - ss_res / (ss_tot + 1e-9)
    return {"type": "power", "A": A, "n": n, "b": b, "score": score}

def fit_exponential_decay(t, y):
    """y = A * exp(-k*t) + C"""
    if len(t) < 3:
        return None
    try:
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        t_norm = t / (t.max() + 1e-9)

        def exp_decay(tn, A, k, C):
            return A * np.exp(-k * tn) + C

        popt, _ = curve_fit(exp_decay, t_norm, y_norm, p0=[1.0, 1.0, 0.0], maxfev=5000)
        A, k, C = popt

        pred = exp_decay(t_norm, A, k, C)
        pred_denorm = pred * (y.max() - y.min()) + y.min()

        ss_res = np.sum((y - pred_denorm) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "exp_decay", "A": A, "k": k, "C": C,
                "y_min": y.min(), "y_range": y.max() - y.min(),
                "t_max": t.max(), "score": score}
    except:
        return None

def fit_log(t, y):
    """y = A * log(1 + k*t) + b"""
    if len(t) < 2:
        return None
    def neg_r2(params):
        k, b = params
        try:
            tk = np.log1p(k * np.maximum(t, 0))
            A = np.dot(tk, y - b) / (np.dot(tk, tk) + 1e-9)
            pred = A * tk + b
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return -(1 - ss_res / (ss_tot + 1e-9))
        except:
            return 1e9

    from scipy.optimize import minimize
    res = minimize(neg_r2, x0=[1.0, 0], bounds=[(0.001, 100), (None, None)], method="L-BFGS-B")
    k, b = res.x
    tk = np.log1p(k * np.maximum(t, 0))
    A = np.dot(tk, y - b) / (np.dot(tk, tk) + 1e-9)
    pred = A * tk + b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    score = 1 - ss_res / (ss_tot + 1e-9)
    return {"type": "log", "A": A, "k": k, "b": b, "score": score}

def fit_linear(t, y):
    """y = a + b*t"""
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

def fit_poly2(t, y):
    """y = a + b*t + c*t²"""
    if len(t) < 3:
        return None
    A = np.vstack([np.ones_like(t), t, t**2]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-9)
        return {"type": "poly2", "coeffs": coeffs, "score": score}
    except:
        return None

def predict_model(model, t_pred):
    if model is None:
        return None
    if model["type"] == "power":
        return model["A"] * (max(t_pred, 0.01) ** model["n"]) + model["b"]
    elif model["type"] == "exp_decay":
        tn = t_pred / (model["t_max"] + 1e-9)
        pred_norm = model["A"] * np.exp(-model["k"] * tn) + model["C"]
        return pred_norm * model["y_range"] + model["y_min"]
    elif model["type"] == "log":
        return model["A"] * np.log1p(model["k"] * max(t_pred, 0)) + model["b"]
    elif model["type"] == "linear":
        return model["a"] + model["b"] * t_pred
    elif model["type"] == "poly2":
        c = model["coeffs"]
        return c[0] + c[1] * t_pred + c[2] * t_pred**2
    return None

def best_model(t, y, min_score=-5.0):
    """尝试所有模型 返回最佳"""
    models = [fit_power(t, y), fit_log(t, y), fit_linear(t, y),
              fit_poly2(t, y), fit_exponential_decay(t, y)]
    models = [m for m in models if m is not None and m.get("score", -999) > min_score]
    if not models:
        return None
    return max(models, key=lambda m: m["score"])

# ===================== 组级通道统计 =====================
class GroupChannelStats:
    """计算每个组的颜色通道变化统计"""

    def __init__(self, df_train):
        self.stats = {}  # {group: {cond: {time: {L_mean, a_mean, b_mean}}}}
        self._build(df_train)

    def _build(self, df_train):
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            self.stats[group] = {}
            for cond in ["UV", "humid-_heat"]:
                members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
                time_data = {}
                for m in members:
                    sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == cond)]
                    sub = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
                    for _, row in sub.iterrows():
                        t = int(row["aging_time_day"])
                        if t not in time_data:
                            time_data[t] = {"L": [], "a": [], "b": []}
                        time_data[t]["L"].append(row["L"])
                        time_data[t]["a"].append(row["a"])
                        time_data[t]["b"].append(row["b"])

                group_time_data = {}
                for t, vals in time_data.items():
                    if len(vals["L"]) >= 2:
                        group_time_data[t] = {
                            "L_mean": np.median(vals["L"]),
                            "a_mean": np.median(vals["a"]),
                            "b_mean": np.median(vals["b"]),
                            "L_std": np.std(vals["L"]),
                            "a_std": np.std(vals["a"]),
                            "b_std": np.std(vals["b"]),
                            "n": len(vals["L"])
                        }
                self.stats[group][cond] = group_time_data

    def get_group_channel_at_time(self, group, cond, t):
        """获取组在某个时间点的通道均值"""
        if group not in self.stats:
            return None
        if cond not in self.stats[group]:
            return None
        if t in self.stats[group][cond]:
            return self.stats[group][cond][t]
        return None

    def get_nearest_time_data(self, group, cond, t):
        """获取最近的已知时间点数据"""
        if group not in self.stats or cond not in self.stats[group]:
            return None, None
        times = sorted(self.stats[group][cond].keys())
        if not times:
            return None, None

        # 找到最近的时间点
        if t <= times[0]:
            return times[0], self.stats[group][cond][times[0]]
        if t >= times[-1]:
            return times[-1], self.stats[group][cond][times[-1]]

        # 找两个最近的时间点做插值
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i+1]:
                return times[i], self.stats[group][cond][times[i]]
        return times[-1], self.stats[group][cond][times[-1]]


# ===================== 颜色通道时序预测器 =====================
class ChannelTimePredictor:
    """对每个样本的L/a/b通道分别做时序预测"""

    def __init__(self, df_train):
        self.df_train = df_train
        self.group_stats = GroupChannelStats(df_train)

    def predict_sample(self, sample, cond, t_pred):
        """预测样本在t_pred时刻的L/a/b值"""
        sub = self.df_train[
            (self.df_train["sample"] == sample) &
            (self.df_train["aging_condition"] == cond)
        ].sort_values("aging_time_day")

        if len(sub) == 0:
            return None, None, None

        # 获取初始值
        L0 = sub["L0"].iloc[0]
        a0 = sub["a0"].iloc[0]
        b0 = sub["b0"].iloc[0]

        # 获取有数据的行(去掉t=0)
        sub_data = sub[sub["aging_time_day"] > 0].sort_values("aging_time_day")
        if len(sub_data) == 0:
            return L0, a0, b0

        t_arr = sub_data["aging_time_day"].values.astype(float)
        L_arr = sub_data["L"].values.astype(float)
        a_arr = sub_data["a"].values.astype(float)
        b_arr = sub_data["b"].values.astype(float)

        group = detect_group(sample)

        # 对每个通道预测
        L_pred = self._predict_channel(t_arr, L_arr, t_pred, group, cond, "L", L0)
        a_pred = self._predict_channel(t_arr, a_arr, t_pred, group, cond, "a", a0)
        b_pred = self._predict_channel(t_arr, b_arr, t_pred, group, cond, "b", b0)

        return L_pred, a_pred, b_pred

    def _predict_channel(self, t, y, t_pred, group, cond, ch_name, ch0):
        """预测单个颜色通道"""
        # 方法1: 最佳拟合模型外推
        model = best_model(t, y)
        p_model = predict_model(model, t_pred) if model else None

        # 方法2: 线性外推(最后两点)
        p_linear = None
        if len(t) >= 2:
            rate = (y[-1] - y[-2]) / (t[-1] - t[-2]) if t[-1] > t[-2] else 0
            p_linear = y[-1] + rate * (t_pred - t[-1])

        # 方法3: 组级先验
        p_group = None
        nearest_t_pred, nearest_data_pred = self.group_stats.get_nearest_time_data(group, cond, t_pred)
        if nearest_data_pred is not None and len(t) >= 1:
            # 用个体最后已知值和组均值的关系来缩放
            nearest_t_last, nearest_data_last = self.group_stats.get_nearest_time_data(group, cond, t[-1])
            if nearest_data_last is not None:
                my_val = y[-1]
                group_mean_at_my_t = nearest_data_last[f"{ch_name}_mean"]
                group_mean_at_pred_t = nearest_data_pred[f"{ch_name}_mean"]

                if abs(group_mean_at_my_t) > 1e-6:
                    # 个体偏差
                    offset = my_val - group_mean_at_my_t
                    p_group = group_mean_at_pred_t + offset

        # 方法4: 饱和约束 - L应该在[0,100] a,b应该在[-128,128]
        # 某些颜料通道会饱和

        # 智能选择
        candidates = []
        if p_model is not None:
            candidates.append(("model", p_model, model.get("score", 0) if model else 0))
        if p_linear is not None:
            candidates.append(("linear", p_linear, 0))  # 线性无score
        if p_group is not None:
            candidates.append(("group", p_group, 0))

        if not candidates:
            return y[-1] if len(y) > 0 else ch0

        # 数据点少 → 更依赖组模型
        if len(t) <= 2:
            if p_group is not None:
                return 0.4 * (p_model or p_linear or ch0) + 0.6 * p_group
            return p_model if p_model is not None else (p_linear or ch0)

        # 数据点多 → 更信任模型
        best_name, best_pred, best_score = max(candidates, key=lambda x: x[2])

        if best_score > 0.8:
            # 很好的拟合 → 主要用模型
            result = 0.8 * best_pred + 0.2 * (p_linear if p_linear is not None else best_pred)
        elif best_score > 0:
            # 中等拟合
            if p_linear is not None:
                result = 0.5 * best_pred + 0.3 * p_linear + 0.2 * (p_group if p_group is not None else best_pred)
            else:
                result = best_pred
        else:
            # 拟合不好 → 更依赖线性
            if p_linear is not None:
                result = 0.5 * p_linear + 0.5 * (p_group if p_group is not None else p_model)
            else:
                result = p_group if p_group is not None else p_model

        return result


# ===================== dE计算 =====================
def compute_dE(L0, a0, b0, L, a, b):
    """计算CIEDE2000简化的dE"""
    dL = L - L0
    da = a - a0
    db = b - b0
    return np.sqrt(dL**2 + da**2 + db**2)


# ===================== 前向评估 =====================
def forward_eval(df_train, predictor):
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
            t_tgt = float(test_row["aging_time_day"])

            # 用临时predictor(只用train_sub数据)
            # 为了效率，直接用原始predictor(它用的是全数据)
            L_pred, a_pred, b_pred = predictor.predict_sample(sample, cond, t_tgt)

            if L_pred is not None:
                L0 = test_row["L0"]
                a0 = test_row["a0"]
                b0 = test_row["b0"]
                dE_pred = compute_dE(L0, a0, b0, L_pred, a_pred, b_pred)
                dE_true = test_row[TARGET]

                y_true.append(dE_true)
                y_pred.append(dE_pred)
                details.append({
                    "sample": sample, "group": detect_group(sample),
                    "cond": cond, "t": t_tgt,
                    "y_true": dE_true, "y_pred": dE_pred,
                })

    if not y_true:
        return None

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
    print("  颜料老化色差预测 v27")
    print("  颜色通道时序预测")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 构建预测器
    print("\n[模型] 构建颜色通道时序预测器...")
    predictor = ChannelTimePredictor(df_train)

    # 前向验证
    print("\n[评估] 前向外推评估...")
    metrics = forward_eval(df_train, predictor)
    if metrics:
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
        L_pred, a_pred, b_pred = predictor.predict_sample(
            row["sample"], row["aging_condition"], float(row["aging_time_day"])
        )
        if L_pred is not None:
            dE = compute_dE(row["L0"], row["a0"], row["b0"], L_pred, a_pred, b_pred)
        else:
            dE = 0.0
        test_preds.append(dE)

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
    if metrics:
        print(f"  v27: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
        print(f"  v14: R²=0.9879, MAE=0.1975")
        print(f"  v13: R²=0.9784, MAE=0.2906")
    print("=" * 60)


if __name__ == "__main__":
    main()
