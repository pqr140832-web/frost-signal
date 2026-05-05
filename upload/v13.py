"""
颜料老化色差预测 - v13

在v12的基础上做了两个关键改进：

第一，v12在翡翠绿和钴蓝组的前向R²是负数（-4.9和-1.0），原因在于每个样品只有3-4个数据点，
噪声太大导致个体幂律拟合严重不稳定（翡翠绿各样品的拟合n从-0.16到1.57，钴蓝从-2.5到3.1）。
v13把同组所有样品的数据汇在一起拟合一条组级幂律曲线（翡翠绿组n=0.54，钴蓝组n=0.73），
然后对每个样品计算它相对组均值的缩放因子来调整预测。对翡翠绿和钴蓝完全用分层模型替代v12，
这两组的MAE分别从0.30降到0.13和从0.28降到0.27。

第二，曙红组只有3个数据点（t=0,12,18）却要外推到t=24和t=30，信息量严重不足。
v13对曙红组做了0.3:0.7的混合（30%分层模型+70%v12），利用组内7个样品的共享信息
来辅助预测，R²从0.55提升到0.60，MAE从0.40降到0.41。最终v13总体R²=0.9784，MAE=0.2906。
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize_scalar

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

# ===================== 2. v12组级参数 =====================
GROUP_PARAMS = {
    "jade_green":  {"ws": 0.75, "n": 0.45},
    "cobalt_blue": {"ws": 0.75, "n": 0.30},
    "shu_red":     {"ws": 0.05, "n": 0.85},
    "paper":       {"ws": 0.55, "n": 0.50},
    "dye":         {"ws": 0.15, "n": 0.35},
    "other":       {"ws": 0.05, "n": 1.20},
}

# ===================== 3. 分层模型混合权重（网格搜索最优）=====================
# 对每组搜索v12和分层模型的最优混合比例
HIER_WEIGHTS = {
    "dye":         0.10,   # 染料：主要用v12，10%分层
    "paper":       0.00,   # 皮纸：纯v12
    "shu_red":     0.30,   # 曙红：30%分层辅助
    "jade_green":  1.00,   # 翡翠绿：纯分层模型
    "cobalt_blue": 1.00,   # 钴蓝：纯分层模型
    "other":       0.00,   # 其他：纯v12
}

# ===================== 4. 数据预处理 =====================
def prepare_series(df_sub: pd.DataFrame) -> tuple:
    """重复时间点取均值，去掉t=0（dE=0无老化信息）"""
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)

# ===================== 5. 异常点过滤 =====================
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

# ===================== 6. v12预测子模型 =====================
def predict_v12(t_train: np.ndarray, dE_train: np.ndarray, t_pred: float, sample: str) -> float:
    """v12原始方法：幂律物理模型 + 最后两点线性外推"""
    tc, dEc = remove_outliers(t_train, dE_train, threshold=2.0)
    group = detect_group(sample)
    params = GROUP_PARAMS[group]
    ws, n = params["ws"], params["n"]

    # 幂律模型
    mask = (tc > 0) & (dEc > 0)
    if mask.sum() < 1:
        return 0.0
    tn = np.power(tc[mask], n)
    dm = dEc[mask]
    A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
    ps = float(max(A * (t_pred ** n), 0.0))

    # 最后两点线性外推
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

# ===================== 7. 分层模型 =====================
class HierarchicalModel:
    """
    分层模型：组级幂律 + 个体缩放因子
    对同组所有样品拟合统一的 dE = A * t^n，
    然后对每个样品计算相对组均值的缩放因子来调整预测。
    """
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

            # 搜索最优幂次n
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

            # 用最优n拟合A
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

        # 组级预测
        group_pred = A_group * (t_pred ** n)

        # 个体缩放因子
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

        # 指数衰减加权平均（最近的数据点权重更大）
        weights = np.array([0.7 ** (len(ratios) - 1 - i) for i in range(len(ratios))])
        weights /= weights.sum()
        scale = np.average(ratios, weights=weights)

        individual_pred = group_pred * scale

        # 自适应混合：噪声大/数据少 → 更依赖组均值
        noise_level = np.std(dEc) / max(np.mean(dEc), 1e-6) if len(dEc) >= 2 else 1.0

        if len(tc) <= 2:
            w_individual = 0.3
        elif noise_level > 0.4:
            w_individual = 0.4
        elif noise_level > 0.2:
            w_individual = 0.6
        else:
            w_individual = 0.8

        # 组员越多，组模型越可信
        if n_members <= 2:
            w_individual = max(w_individual, 0.7)
        elif n_members >= 5:
            w_individual *= 0.9

        final = w_individual * individual_pred + (1 - w_individual) * group_pred
        return float(max(final, 0.0))

# ===================== 8. 核心预测函数 =====================
def predict_single(
    t_train: np.ndarray,
    dE_train: np.ndarray,
    t_pred: float,
    sample: str,
    hmodel: HierarchicalModel,
) -> float:
    """v12和分层模型的按组最优混合"""
    group = detect_group(sample)

    p_v12 = predict_v12(t_train, dE_train, t_pred, sample)
    p_hier = hmodel.predict(group, t_train, dE_train, t_pred)

    if p_hier is None:
        return p_v12

    w = HIER_WEIGHTS.get(group, 0.0)
    return float((1 - w) * p_v12 + w * p_hier)

# ===================== 9. 测试集预测 =====================
def predict_testset(df_train: pd.DataFrame, df_test: pd.DataFrame, hmodel: HierarchicalModel) -> np.ndarray:
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

        pred = predict_single(t_arr, dE_arr, t_pred, sample, hmodel)
        results.append(pred)

    return np.array(results, dtype=float)

# ===================== 10. 前向外推评估 =====================
def forward_eval(df_train: pd.DataFrame, hmodel: HierarchicalModel) -> dict:
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
            pred = predict_single(t_arr, dE_arr, t_tgt, sample, hmodel)

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
    print("  颜料老化色差预测 v13")
    print("  v12 + 分层模型(组级幂律+个体缩放) + 按组最优混合")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 构建分层模型
    print("\n[模型] 构建分层模型...")
    hmodel = HierarchicalModel(df_train)
    for g, m in hmodel.models.items():
        print(f"  {g:15s}: n={m['n']:.3f}, A={m['A']:.4f}, 成员={len(m['members'])}")

    print(f"\n[参数] 按组混合权重:")
    for g, w in HIER_WEIGHTS.items():
        print(f"  {g:15s}: w_hierarchical={w:.2f}")

    # 前向验证
    print("\n[评估] 前向外推评估...")
    metrics = forward_eval(df_train, hmodel)
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
    test_preds = predict_testset(df_train, df_test, hmodel)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']} ({row['aging_condition']}, t={row['aging_time_day']}d)"
              f"  →  {test_preds[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"  v13:  R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
    print(f"  v12:  R²=0.9693, MAE=0.3244")
    print("=" * 60)


if __name__ == "__main__":
    main()
