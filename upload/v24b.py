"""
颜料老化色差预测 - v24b

核心策略：v13为底座，只做两个确定性改进：
1. 曙红组用多方法中位数 + 组级保守收缩（38%测试点的决定性改进）
2. 保留v13的所有参数（GROUP_PARAMS, HIER_WEIGHTS），不做参数搜索

v13已验证的参数是最优的（61.9分），不应该重新搜索。
问题在于曙红组的前向评估误差太大。
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
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

# ===================== 2. v13组级参数（保留原值）=====================
GROUP_PARAMS = {
    "jade_green":  {"ws": 0.75, "n": 0.45},
    "cobalt_blue": {"ws": 0.75, "n": 0.30},
    "shu_red":     {"ws": 0.05, "n": 0.85},
    "paper":       {"ws": 0.55, "n": 0.50},
    "dye":         {"ws": 0.15, "n": 0.35},
    "other":       {"ws": 0.05, "n": 1.20},
}

HIER_WEIGHTS = {
    "dye":         0.10,
    "paper":       0.00,
    "shu_red":     0.30,
    "jade_green":  1.00,
    "cobalt_blue": 1.00,
    "other":       0.00,
}

# ===================== 3. 数据预处理 =====================
def prepare_series(df_sub: pd.DataFrame) -> tuple:
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)

# ===================== 4. 异常点过滤 =====================
def remove_outliers(t: np.ndarray, dE: np.ndarray, threshold: float = 2.5) -> tuple:
    t = np.asarray(t, dtype=float)
    dE = np.asarray(dE, dtype=float)
    if len(t) < 3:
        return t.copy(), dE.copy()
    diffs = np.diff(dE)
    mean_abs = np.mean(np.abs(diffs))
    if mean_abs < 1e-6:
        return t, dE
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t) - 1):
        if abs(diffs[i - 1]) > threshold * mean_abs:
            keep[i] = False
    return t[keep], dE[keep]

# ===================== 5. v12预测子模型（原v13逻辑）=====================
def predict_v12(t_train: np.ndarray, dE_train: np.ndarray, t_pred: float, sample: str) -> float:
    tc, dEc = remove_outliers(t_train, dE_train, threshold=2.0)
    group = detect_group(sample)
    params = GROUP_PARAMS[group]
    ws, n = params["ws"], params["n"]

    mask = (tc > 0) & (dEc > 0)
    if mask.sum() < 1:
        return 0.0
    tn = np.power(tc[mask], n)
    dm = dEc[mask]
    A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
    ps = float(max(A * (t_pred ** n), 0.0))

    if len(tc) < 2:
        pl = float(max(dEc[0] / tc[0] * t_pred, 0.0)) if tc[0] > 0 else 0.0
    else:
        t1, t2 = tc[-2], tc[-1]
        d1, d2 = dEc[-2], dEc[-1]
        if t2 <= t1:
            pl = float(max(d2, 0.0))
        else:
            rate = (d2 - d1) / (t2 - t1)
            pl = float(max(d2 + rate * (t_pred - t2), 0.0))

    return float(max(ws * ps + (1 - ws) * pl, 0.0))

# ===================== 6. 分层模型（原v13逻辑）=====================
class HierarchicalModel:
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

    def predict(self, group: str, t_train: np.ndarray, dE_train: np.ndarray, t_pred: float) -> float:
        if group not in self.models:
            return None
        model = self.models[group]
        n, A_group = model["n"], model["A"]
        n_members = len(model["members"])

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

        if n_members <= 2:
            w_individual = max(w_individual, 0.7)
        elif n_members >= 5:
            w_individual *= 0.9

        final = w_individual * individual_pred + (1 - w_individual) * group_pred
        return float(max(final, 0))


# ===================== 7. 曙红组强化预测 =====================
class ShuRedEnhanced:
    """
    曙红组：7个样品，每个只有t=12,18两个非零点
    测试点：t=24(7个) + t=30(7个) = 14个，占总数38%

    改进策略：
    - 多方法预测取中位数：线性外推、幂律、log模型
    - 保守收缩：向组均值外推结果收缩
    - 增长率递减：12→18的增长率在18→24应该减半
    """
    def __init__(self, df_train: pd.DataFrame):
        self.group_stats = {}
        self._build(df_train)

    def _build(self, df_train):
        members = [s for s in df_train["sample"].unique() if detect_group(s) == "shu_red"]
        time_dE = {}
        for m in members:
            sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == "UV")]
            sub = sub.sort_values("aging_time_day")
            for _, row in sub.iterrows():
                t = row["aging_time_day"]
                if t > 0:
                    time_dE.setdefault(t, []).append(row[TARGET])

        for t, vals in sorted(time_dE.items()):
            self.group_stats[t] = {
                "median": np.median(vals),
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals),
            }

        # 组增长率
        if 12 in self.group_stats and 18 in self.group_stats:
            self.growth_rate_12_18 = (self.group_stats[18]["median"] - self.group_stats[12]["median"]) / 6
        else:
            self.growth_rate_12_18 = 0

    def predict(self, t_train, dE_train, t_pred):
        tc = np.asarray(t_train, dtype=float)
        dEc = np.asarray(dE_train, dtype=float)

        preds = []

        # 方法1: 最后两点线性外推
        if len(tc) >= 2:
            t1, t2 = tc[-2], tc[-1]
            d1, d2 = dEc[-2], dEc[-1]
            if t2 > t1:
                rate = (d2 - d1) / (t2 - t1)
                preds.append(max(d2 + rate * (t_pred - t2), 0))
            else:
                preds.append(max(d2, 0))

        # 方法2: 幂律拟合
        if len(tc) >= 2:
            mask = (tc > 0) & (dEc > 0)
            if mask.sum() >= 2:
                def neg_r2(n):
                    tn = np.power(tc[mask], n)
                    A = np.dot(tn, dEc[mask]) / (np.dot(tn, tn) + 1e-9)
                    pred = A * tn
                    ss_res = np.sum((dEc[mask] - pred) ** 2)
                    ss_tot = np.sum((dEc[mask] - dEc[mask].mean()) ** 2)
                    return -(1 - ss_res / (ss_tot + 1e-9))
                res = minimize_scalar(neg_r2, bounds=(0.2, 1.5), method="bounded")
                n = res.x
                tn = np.power(tc[mask], n)
                A = np.dot(tn, dEc[mask]) / (np.dot(tn, tn) + 1e-9)
                preds.append(max(A * (t_pred ** n), 0))

        # 方法3: 减速增长率模型
        if self.growth_rate_12_18 > 0 and len(tc) >= 1:
            last_t = tc[-1]
            last_dE = dEc[-1]
            rate = self.growth_rate_12_18
            # 每经过6天增长率衰减50%
            periods = (t_pred - last_t) / 6
            if periods <= 1:
                factor = 1.0
            else:
                factor = 0.5 ** (periods - 1)
            p_decay = last_dE + rate * (t_pred - last_t) * factor
            preds.append(max(p_decay, 0))

        # 方法4: log模型
        if len(tc) >= 2:
            mask = (tc > 0) & (dEc > 0)
            if mask.sum() >= 2:
                def neg_r2_log(logk):
                    k = np.exp(logk)
                    tk = np.log(1 + k * tc[mask])
                    A = np.dot(tk, dEc[mask]) / (np.dot(tk, tk) + 1e-9)
                    pred = A * tk
                    ss_res = np.sum((dEc[mask] - pred) ** 2)
                    ss_tot = np.sum((dEc[mask] - dEc[mask].mean()) ** 2)
                    return -(1 - ss_res / (ss_tot + 1e-9))
                res = minimize_scalar(neg_r2_log, bounds=(-3, 2), method="bounded")
                k = np.exp(res.x)
                tk = np.log(1 + k * tc[mask])
                A = np.dot(tk, dEc[mask]) / (np.dot(tk, tk) + 1e-9)
                p_log = A * np.log(1 + k * t_pred)
                preds.append(max(p_log, 0))

        if not preds:
            if 18 in self.group_stats:
                return self.group_stats[18]["median"]
            return 1.0

        # 取中位数
        result = float(np.median(preds))

        # 保守收缩：向组均值外推收缩
        if 12 in self.group_stats and 18 in self.group_stats:
            group_12 = self.group_stats[12]["median"]
            group_18 = self.group_stats[18]["median"]
            # 组级外推：用减速增长率
            group_rate = self.growth_rate_12_18
            if t_pred <= 24:
                group_extrap = group_18 + group_rate * (t_pred - 18)
            else:
                group_extrap = group_18 + group_rate * 6 * 0.5 + group_rate * 0.5 * (t_pred - 24)
            group_extrap = max(group_extrap, 0)

            # 混合：70%多方法中位数 + 30%组级外推
            result = 0.7 * result + 0.3 * group_extrap

        return float(max(result, 0))


# ===================== 8. 核心预测函数 =====================
def predict_single(t_train, dE_train, t_pred, sample, hmodel, shu_red_model):
    group = detect_group(sample)

    # 曙红组用强化模型
    if group == "shu_red":
        return shu_red_model.predict(t_train, dE_train, t_pred)

    # 其他组保持v13原逻辑
    p_v12 = predict_v12(t_train, dE_train, t_pred, sample)
    p_hier = hmodel.predict(group, t_train, dE_train, t_pred)

    if p_hier is None:
        return p_v12

    w = HIER_WEIGHTS.get(group, 0.0)
    return float((1 - w) * p_v12 + w * p_hier)


# ===================== 9. 测试集预测 =====================
def predict_testset(df_train, df_test, hmodel, shu_red_model):
    results = []
    for _, row in df_test.iterrows():
        sample = row["sample"]
        cond = row["aging_condition"]
        t_pred = float(row["aging_time_day"])

        sub = df_train[
            (df_train["sample"] == sample) &
            (df_train["aging_condition"] == cond)
        ].sort_values("aging_time_day")

        if len(sub) == 0:
            results.append(float(df_train[TARGET].mean()))
            continue

        t_arr, dE_arr = prepare_series(sub)
        if len(t_arr) == 0:
            results.append(0.0)
            continue

        pred = predict_single(t_arr, dE_arr, t_pred, sample, hmodel, shu_red_model)
        results.append(pred)

    return np.array(results, dtype=float)


# ===================== 10. 前向外推评估 =====================
def forward_eval(df_train, hmodel, shu_red_model):
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
            pred = predict_single(t_arr, dE_arr, t_tgt, sample, hmodel, shu_red_model)

            y_true.append(float(test_row[TARGET]))
            y_pred.append(pred)
            details.append({
                "sample": sample,
                "group": detect_group(sample),
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
    print("  颜料老化色差预测 v24b")
    print("  v13底座 + 曙红强化(多方法中位数+保守收缩)")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 构建分层模型（和v13完全一样）
    print("\n[模型] 构建分层模型...")
    hmodel = HierarchicalModel(df_train)
    for g, m in hmodel.models.items():
        print(f"  {g:15s}: n={m['n']:.3f}, A={m['A']:.4f}, 成员={len(m['members'])}")

    # 构建曙红强化模型
    print("\n[模型] 构建曙红强化模型...")
    shu_red_model = ShuRedEnhanced(df_train)
    print(f"  组中位数: t=12→{shu_red_model.group_stats.get(12, {}).get('median', 0):.3f}, "
          f"t=18→{shu_red_model.group_stats.get(18, {}).get('median', 0):.3f}")
    print(f"  增长率(12→18): {shu_red_model.growth_rate_12_18:.4f}/day")

    # 前向验证
    print("\n[评估] 前向外推评估...")
    metrics = forward_eval(df_train, hmodel, shu_red_model)
    print(f"  有效序列数: {metrics['n']}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")

    # 按组评估
    print("\n[评估] 按组详细:")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        gdetails = [d for d in metrics["details"] if d["group"] == g]
        if len(gdetails) >= 2:
            yt = np.array([d["y_true"] for d in gdetails])
            yp = np.array([d["y_pred"] for d in gdetails])
            r2 = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            print(f"  {g:15s}: R²={r2:+.4f}, MAE={mae:.4f}, n={len(gdetails)}")

    # 测试集预测
    print("\n[预测] 生成测试集预测...")
    test_preds = predict_testset(df_train, df_test, hmodel, shu_red_model)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    # 对比v13
    print("\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d)"
              f"  →  {test_preds[i]:.4f}")

    print("\n[对比]")
    print(f"  v24b: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
    print(f"  v13:  R²=0.9784, MAE=0.2906 (score=61.9)")

    # v13的前向评估（用v13原始预测对比）
    # 手动跑一遍v13前向评估看曙红组
    print("\n[曙红组详细]")
    shu_details = [d for d in metrics["details"] if d["group"] == "shu_red"]
    for d in shu_details:
        err = abs(d["y_pred"] - d["y_true"])
        print(f"  {d['sample']:20s}: true={d['y_true']:.3f}, pred={d['y_pred']:.3f}, err={err:.3f}")


if __name__ == "__main__":
    main()
