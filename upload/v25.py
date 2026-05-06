"""
颜料老化色差预测 - v25

关键洞察：v13的forward eval和实际提交之间有一个巨大的鸿沟
v13 forward eval MAE=0.29但score=61.9，说明metric可能不是简单MAE

v25的核心改进：针对测试集外推距离做更精准的建模
- 染料：t=4→5（外推25%），v13应该已经很准
- 皮纸：t=15→40（外推167%），距离太远
- 曙红：t=18→24(33%) 和 t=18→30(67%)
- 翡翠绿/钴蓝：t=24→30(25%)

最确定的改进方向：
1. 染料组(v13 MAE=0.39，最大)：改进v12参数
2. other组(v13 MAE=0.38)：改进参数
3. 皮纸t=40远距离外推：加饱和保护
4. 曙红t=30远距离外推：加增长率衰减

本版本：逐组精确优化v12的(ws,n)参数，用LOPO-CV搜索
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = SCRIPT_DIR / "paint_aging_trainset.csv"
TEST_CSV  = SCRIPT_DIR / "paint_aging_testset.csv"
OUT_CSV   = SCRIPT_DIR / "predict_out.csv"
TARGET = "dietaE"

def detect_group(sample: str) -> str:
    if "翡翠绿" in sample: return "jade_green"
    if "钴蓝"  in sample: return "cobalt_blue"
    if "曙红"  in sample: return "shu_red"
    if "皮纸"  in sample: return "paper"
    if any(x in sample for x in ["染料", "紫草", "苏木", "红花", "黄檗"]):
        return "dye"
    return "other"

def prepare_series(df_sub: pd.DataFrame) -> tuple:
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)

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


def search_best_params_and_weights(df_train, hmodel):
    """
    对每组搜索最优v12(ws,n)和混合权重w
    用LOPO-CV评估
    """
    results = {}

    for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
        all_cv_data = []

        for m in members:
            for cond in df_train[df_train["sample"] == m]["aging_condition"].unique():
                sub = df_train[
                    (df_train["sample"] == m) &
                    (df_train["aging_condition"] == cond)
                ].sort_values("aging_time_day")

                t_all, dE_all = prepare_series(sub)
                if len(t_all) < 3:
                    continue

                t_train, dE_train = t_all[:-1], dE_all[:-1]
                t_test, dE_test = t_all[-1], dE_all[-1]

                tc, dEc = remove_outliers(t_train, dE_train, threshold=2.0)
                all_cv_data.append((tc, dEc, t_test, dE_test, m))

        if not all_cv_data:
            continue

        best_mae = 999
        best_ws, best_n, best_w = 0.5, 0.5, 0.0

        # 精细网格搜索
        for n in np.arange(0.10, 1.51, 0.05):
            for ws in np.arange(0.00, 1.01, 0.05):
                errors = []
                for tc, dEc, t_test, dE_test, m in all_cv_data:
                    # v12 prediction with these params
                    mask = (tc > 0) & (dEc > 0)
                    if mask.sum() < 1:
                        continue
                    tn = np.power(tc[mask], n)
                    dm = dEc[mask]
                    A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
                    ps = float(max(A * (t_test ** n), 0.0))

                    if len(tc) < 2:
                        pl = float(max(dEc[0] / tc[0] * t_test, 0.0)) if tc[0] > 0 else 0.0
                    else:
                        t1, t2 = tc[-2], tc[-1]
                        d1, d2 = dEc[-2], dEc[-1]
                        if t2 <= t1:
                            pl = float(max(d2, 0.0))
                        else:
                            rate = (d2 - d1) / (t2 - t1)
                            pl = float(max(d2 + rate * (t_test - t2), 0.0))

                    p_v12 = float(max(ws * ps + (1 - ws) * pl, 0.0))
                    errors.append((p_v12, dE_test))

                if not errors:
                    continue

                # 搜索最优混合权重
                p_v12_arr = np.array([e[0] for e in errors])
                dE_test_arr = np.array([e[1] for e in errors])

                for w_hier_frac in np.arange(0.0, 1.01, 0.10):
                    mae_sum = 0
                    for idx, (tc, dEc, t_test, dE_test, m) in enumerate(all_cv_data):
                        p_v12 = p_v12_arr[idx]
                        p_hier = hmodel.predict(group, tc, dEc, t_test)
                        if p_hier is None:
                            pred = p_v12
                        else:
                            pred = (1 - w_hier_frac) * p_v12 + w_hier_frac * p_hier
                        mae_sum += abs(pred - dE_test)

                    total_mae = mae_sum / len(errors)
                    if total_mae < best_mae:
                        best_mae = total_mae
                        best_ws = ws
                        best_n = n
                        best_w = w_hier_frac

        results[group] = {
            "ws": best_ws, "n": best_n, "w_hier": best_w, "cv_mae": best_mae
        }

    return results


def predict_v12_with_params(t_train, dE_train, t_pred, ws, n):
    """v12 prediction with given params"""
    tc, dEc = remove_outliers(t_train, dE_train, threshold=2.0)

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


def predict_single(t_train, dE_train, t_pred, sample, hmodel, optimized_params):
    group = detect_group(sample)
    params = optimized_params.get(group)

    if params is None:
        return 0.0

    p_v12 = predict_v12_with_params(t_train, dE_train, t_pred, params["ws"], params["n"])
    p_hier = hmodel.predict(group, t_train, dE_train, t_pred)

    if p_hier is None:
        return p_v12

    w = params["w_hier"]
    return float((1 - w) * p_v12 + w * p_hier)


def predict_testset(df_train, df_test, hmodel, optimized_params):
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

        pred = predict_single(t_arr, dE_arr, t_pred, sample, hmodel, optimized_params)
        results.append(pred)

    return np.array(results, dtype=float)


def forward_eval(df_train, hmodel, optimized_params):
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
            pred = predict_single(t_arr, dE_arr, t_tgt, sample, hmodel, optimized_params)

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


def main():
    print("=" * 60)
    print("  颜料老化色差预测 v25")
    print("  全参数LOPO-CV搜索: (ws, n, w_hier)")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    print("\n[模型] 构建分层模型...")
    hmodel = HierarchicalModel(df_train)
    for g, m in hmodel.models.items():
        print(f"  {g:15s}: n={m['n']:.3f}, A={m['A']:.4f}, 成员={len(m['members'])}")

    print("\n[搜索] LOPO-CV搜索每组(ws, n, w_hier)...")
    v13_params = {
        "dye": {"ws": 0.15, "n": 0.35, "w_hier": 0.10},
        "paper": {"ws": 0.55, "n": 0.50, "w_hier": 0.00},
        "shu_red": {"ws": 0.05, "n": 0.85, "w_hier": 0.30},
        "jade_green": {"ws": 0.75, "n": 0.45, "w_hier": 1.00},
        "cobalt_blue": {"ws": 0.75, "n": 0.30, "w_hier": 1.00},
        "other": {"ws": 0.05, "n": 1.20, "w_hier": 0.00},
    }

    optimized_params = search_best_params_and_weights(df_train, hmodel)
    for g, p in optimized_params.items():
        v13p = v13_params.get(g, {})
        print(f"  {g:15s}: ws={p['ws']:.2f}(v13={v13p.get('ws',0):.2f}), "
              f"n={p['n']:.2f}(v13={v13p.get('n',0):.2f}), "
              f"w_hier={p['w_hier']:.2f}(v13={v13p.get('w_hier',0):.2f}), "
              f"CV_MAE={p['cv_mae']:.4f}")

    # 前向验证
    print("\n[评估] v25前向外推评估...")
    metrics = forward_eval(df_train, hmodel, optimized_params)
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")

    print("\n[评估] 按组:")
    for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
        gd = [d for d in metrics["details"] if d["group"] == g]
        if len(gd) >= 2:
            yt = np.array([d["y_true"] for d in gd])
            yp = np.array([d["y_pred"] for d in gd])
            print(f"  {g:15s}: MAE={mean_absolute_error(yt, yp):.4f}")

    # 测试集预测
    print("\n[预测] 生成测试集预测...")
    test_preds = predict_testset(df_train, df_test, hmodel, optimized_params)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d)"
              f"  →  {test_preds[i]:.4f}")

    print(f"\n[对比] v25: MAE={metrics['MAE']:.4f}, v13: MAE=0.2906")


if __name__ == "__main__":
    main()
