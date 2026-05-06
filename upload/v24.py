"""
颜料老化色差预测 - v24

基于v13的确定性改进版本，三个核心改动：

1. 曙红组强化：曙红有14个测试点(占38%)但只有2个非零训练点(12,18天)
   → 改用组级log模型 + 个体缩放 + 保守向组均值收缩
   → 对t=24和t=30分别用不同策略：t=24较近用线性+组级，t=30较远用log饱和

2. 参数自动优化：v12的组参数(ws,n)和混合权重不再hardcode
   → 用Leave-One-Point-Out交叉验证自动搜索每组最优(ws,n)和混合权重
   → 曙红和翡翠绿/钴蓝用分层模型的最优参数

3. 远距离外推保护：对所有组的长距离外推增加饱和保护
   → 幂律外推不超过组内观测最大值的2倍增长率
   → 对dye组(t=4→5外推距离短)不加限制
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

# ===================== 2. 数据预处理 =====================
def prepare_series(df_sub: pd.DataFrame) -> tuple:
    """重复时间点取均值，去掉t=0（dE=0无老化信息）"""
    agg = df_sub.groupby("aging_time_day").agg({TARGET: "mean"}).reset_index()
    agg = agg[agg["aging_time_day"] > 0].sort_values("aging_time_day")
    return agg["aging_time_day"].values.astype(float), agg[TARGET].values.astype(float)

# ===================== 3. 异常点过滤 =====================
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

# ===================== 4. 幂律拟合 =====================
def fit_power_law(t: np.ndarray, dE: np.ndarray) -> dict:
    """拟合 dE = A * t^n，返回{n, A, score}"""
    mask = (t > 0) & (dE > 0)
    if mask.sum() < 2:
        return None
    tn = np.power(t[mask], 0.5)
    dm = dE[mask]

    def neg_r2(n):
        tn2 = np.power(t[mask], n)
        A = np.dot(tn2, dm) / (np.dot(tn2, tn2) + 1e-9)
        pred = A * tn2
        ss_res = np.sum((dm - pred) ** 2)
        ss_tot = np.sum((dm - dm.mean()) ** 2)
        return -(1 - ss_res / (ss_tot + 1e-9))

    result = minimize_scalar(neg_r2, bounds=(0.1, 2.0), method="bounded")
    best_n = result.x
    tn = np.power(t[mask], best_n)
    A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
    return {"n": best_n, "A": A, "score": -result.fun}

# ===================== 5. 分层模型（改进版）=====================
class HierarchicalModel:
    """
    改进的分层模型：
    - 组级幂律自动搜索最优n
    - 个体缩放因子用指数衰减加权
    - 自适应混合权重
    - 远距离外推保护
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

            # 记录组内最大dE和对应时间（用于远距离外推保护）
            max_dE = all_dE.max()
            max_t = all_t[all_dE.argmax()]

            self.models[group] = {
                "n": best_n, "A": A, "members": members,
                "max_dE": max_dE, "max_t": max_t,
                "all_t": all_t, "all_dE": all_dE,
            }

    def predict(self, group: str, t_train: np.ndarray, dE_train: np.ndarray,
                t_pred: float) -> float:
        if group not in self.models:
            return None

        model = self.models[group]
        n, A_group = model["n"], model["A"]
        n_members = len(model["members"])

        # 组级预测
        group_pred = A_group * (t_pred ** n)

        # ★ 远距离外推保护：如果t_pred远大于组内最大时间，限制增长速率
        max_t = model["max_t"]
        max_dE = model["max_dE"]
        if t_pred > max_t and max_t > 0:
            # 在max_t处的组级预测值
            group_at_max = A_group * (max_t ** n)
            if group_at_max > 1e-6:
                # 从max_t到t_pred的增长倍数，限制不超过观测增长率的趋势
                obs_rate = max_dE / group_at_max  # 实际/预测 比值
                # 如果外推过远，用线性趋势限制
                extrapolation_factor = t_pred / max_t
                if extrapolation_factor > 1.5:
                    # 用对数衰减限制增长
                    dampened = group_at_max * (1 + np.log(extrapolation_factor))
                    group_pred = min(group_pred, max(dampened, group_at_max))

        group_pred = max(group_pred, 0)

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

        # ★ 保守预测：数据少→更依赖组均值
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
        return float(max(final, 0))


# ===================== 6. v12子模型（自动优化参数）=====================
class V12AutoOpt:
    """
    自动优化每组v12的(ws, n)参数
    用Leave-One-Point-Out CV找到最优参数
    """
    def __init__(self, df_train: pd.DataFrame):
        self.group_params = {}  # {group: {ws, n}}
        self._optimize_all(df_train)

    def _eval_params(self, ws, n, df_train, group):
        """用LOPO-CV评估一组(ws,n)参数在指定组的MAE"""
        errors = []
        members = [s for s in df_train["sample"].unique() if detect_group(s) == group]

        for m in members:
            for cond in df_train[df_train["sample"] == m]["aging_condition"].unique():
                sub = df_train[
                    (df_train["sample"] == m) &
                    (df_train["aging_condition"] == cond)
                ].sort_values("aging_time_day")

                t_all, dE_all = prepare_series(sub)
                if len(t_all) < 3:
                    continue

                # 留最后一个点测试
                t_train, dE_train = t_all[:-1], dE_all[:-1]
                t_test, dE_test = t_all[-1], dE_all[-1]

                tc, dEc = remove_outliers(t_train, dE_train, threshold=2.0)

                # 幂律
                mask = (tc > 0) & (dEc > 0)
                if mask.sum() < 1:
                    continue
                tn = np.power(tc[mask], n)
                dm = dEc[mask]
                A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
                ps = float(max(A * (t_test ** n), 0.0))

                # 最后两点线性
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

                pred = float(max(ws * ps + (1 - ws) * pl, 0.0))
                errors.append(abs(pred - dE_test))

        if not errors:
            return 999.0
        return np.mean(errors)

    def _optimize_all(self, df_train: pd.DataFrame):
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            best_mae = 999
            best_ws, best_n = 0.5, 0.5

            # 网格搜索
            for n in np.arange(0.1, 1.8, 0.1):
                for ws in np.arange(0.0, 1.0, 0.05):
                    mae = self._eval_params(ws, n, df_train, group)
                    if mae < best_mae:
                        best_mae = mae
                        best_ws, best_n = ws, n

            self.group_params[group] = {"ws": best_ws, "n": best_n, "cv_mae": best_mae}

    def predict(self, t_train, dE_train, t_pred, group):
        """v12预测"""
        if group not in self.group_params:
            return None
        params = self.group_params[group]
        ws, n = params["ws"], params["n"]

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


# ===================== 7. 曙红组专用模型 =====================
class ShuRedModel:
    """
    曙红组专用：7个样品，每个只有t=12,18两个非零点
    需要预测t=24和t=30（14个测试点，占总数38%）

    策略：
    - t=24: 距离t=18较近(6天)，用最后两点线性外推 + 组级约束
    - t=30: 距离较远(12天)，用log模型 + 更强组级约束
    - 最终：个体预测 × 0.6 + 组均值预测 × 0.4（保守收缩）
    """
    def __init__(self, df_train: pd.DataFrame):
        self.group_median_dE = {}  # {t: median dE across group}
        self.group_growth_rate = None
        self._build(df_train)

    def _build(self, df_train: pd.DataFrame):
        # 计算每个时间点的组中位数dE
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
            self.group_median_dE[t] = np.median(vals)

        # 组平均增长率
        if 12 in self.group_median_dE and 18 in self.group_median_dE:
            rate = (self.group_median_dE[18] - self.group_median_dE[12]) / (18 - 12)
            self.group_growth_rate = rate

    def predict(self, t_train, dE_train, t_pred, sample, df_train):
        """预测曙红组"""
        tc = np.array(t_train, dtype=float)
        dEc = np.array(dE_train, dtype=float)

        # 方法1: 最后两点线性外推
        if len(tc) >= 2:
            t1, t2 = tc[-2], tc[-1]
            d1, d2 = dEc[-2], dEc[-1]
            if t2 > t1:
                rate = (d2 - d1) / (t2 - t1)
                p_linear = max(d2 + rate * (t_pred - t2), 0)
            else:
                p_linear = max(d2, 0)
        elif len(tc) >= 1:
            # 用组增长率
            if self.group_growth_rate is not None:
                p_linear = max(dEc[0] + self.group_growth_rate * (t_pred - tc[0]), 0)
            else:
                p_linear = max(dEc[0], 0)
        else:
            p_linear = 0

        # 方法2: 组级趋势外推
        p_group = 0
        if len(tc) >= 1:
            # 找到最近的组中位数时间点，计算个体/组均值比例
            last_t = tc[-1]
            last_dE = dEc[-1]
            closest_t = min(self.group_median_dE.keys(), key=lambda x: abs(x - last_t))
            group_dE_at_t = self.group_median_dE[closest_t]

            if group_dE_at_t > 1e-6 and abs(closest_t - last_t) <= 6:
                scale = last_dE / group_dE_at_t

                # 用组增长率外推
                if self.group_growth_rate is not None:
                    p_group = max(group_dE_at_t + self.group_growth_rate * (t_pred - closest_t), 0) * scale

                # 也用log趋势：dE增长应该减速
                # 曙红组t=12→18增长，t=24,30增长应该放缓
                if 12 in self.group_median_dE and 18 in self.group_median_dE:
                    dE_12 = self.group_median_dE[12]
                    dE_18 = self.group_median_dE[18]
                    # log模型: dE = a * ln(1 + b*t)
                    # 从两个点反推参数
                    if dE_18 > dE_12 and dE_12 > 0:
                        # dE_18/dE_12 = ln(1+18b)/ln(1+12b)
                        # 简化：用线性衰减增长率
                        rate_12_18 = (dE_18 - dE_12) / 6
                        # 增长率每6天衰减30%
                        if t_pred <= 18:
                            decayed_rate = rate_12_18
                        elif t_pred <= 24:
                            decayed_rate = rate_12_18 * 0.7
                        else:
                            decayed_rate = rate_12_18 * 0.49  # 0.7^2
                        p_log = max(dE_18 + decayed_rate * (t_pred - 18), 0) * scale
                        p_group = 0.5 * p_group + 0.5 * p_log if p_group > 0 else p_log

        # 方法3: 幂律模型
        p_power = 0
        if len(tc) >= 2:
            mask = (tc > 0) & (dEc > 0)
            if mask.sum() >= 2:
                from scipy.optimize import minimize_scalar
                def neg_r2(n):
                    tn = np.power(tc[mask], n)
                    A = np.dot(tn, dEc[mask]) / (np.dot(tn, tn) + 1e-9)
                    pred = A * tn
                    ss_res = np.sum((dEc[mask] - pred) ** 2)
                    ss_tot = np.sum((dEc[mask] - dEc[mask].mean()) ** 2)
                    return -(1 - ss_res / (ss_tot + 1e-9))
                res = minimize_scalar(neg_r2, bounds=(0.1, 2.0), method="bounded")
                n = res.x
                tn = np.power(tc[mask], n)
                A = np.dot(tn, dEc[mask]) / (np.dot(tn, tn) + 1e-9)
                p_power = max(A * (t_pred ** n), 0)

        # 融合策略
        preds = []
        if p_linear > 0:
            preds.append(p_linear)
        if p_group > 0:
            preds.append(p_group)
        if p_power > 0:
            preds.append(p_power)

        if not preds:
            # 回退：用组均值
            if 18 in self.group_median_dE:
                return self.group_median_dE[18] * 1.1  # 微小增长
            return 1.0

        # 多方法取中位数（比平均更鲁棒）
        final = float(np.median(preds))

        # 保守收缩：向组均值靠近
        if self.group_growth_rate is not None and 18 in self.group_median_dE:
            group_at_18 = self.group_median_dE[18]
            # 从t=18用组增长率推算
            group_extrapolated = max(group_at_18 + self.group_growth_rate * (t_pred - 18), 0)
            # 混合：60%个体预测 + 40%组级外推
            final = 0.6 * final + 0.4 * group_extrapolated

        return float(max(final, 0))


# ===================== 8. 混合权重自动优化 =====================
class WeightOptimizer:
    """用LOPO-CV优化每组v12和分层模型的混合权重"""
    def __init__(self, df_train, hmodel, v12model, shu_red_model):
        self.weights = {}
        self._optimize(df_train, hmodel, v12model, shu_red_model)

    def _optimize(self, df_train, hmodel, v12model, shu_red_model):
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            errors_v12 = []
            errors_hier = []
            errors_shu = []

            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]

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

                    p_v12 = v12model.predict(t_train, dE_train, t_test, group)
                    p_hier = hmodel.predict(group, t_train, dE_train, t_test)

                    if p_v12 is not None:
                        errors_v12.append(abs(p_v12 - dE_test))
                    if p_hier is not None:
                        errors_hier.append(abs(p_hier - dE_test))

                    if group == "shu_red":
                        p_shu = shu_red_model.predict(t_train, dE_train, t_test, m, df_train)
                        if p_shu is not None:
                            errors_shu.append(abs(p_shu - dE_test))

            if group == "shu_red" and errors_shu:
                # 曙红组有专用模型
                shu_mae = np.mean(errors_shu)
                v12_mae = np.mean(errors_v12) if errors_v12 else 999
                hier_mae = np.mean(errors_hier) if errors_hier else 999

                if shu_mae <= min(v12_mae, hier_mae):
                    self.weights[group] = "shu_red_special"
                elif hier_mae <= v12_mae:
                    self.weights[group] = "hier"
                else:
                    self.weights[group] = "v12"
            else:
                v12_mae = np.mean(errors_v12) if errors_v12 else 999
                hier_mae = np.mean(errors_hier) if errors_hier else 999

                if hier_mae < v12_mae * 0.9:
                    self.weights[group] = "hier"
                elif v12_mae < hier_mae * 0.9:
                    self.weights[group] = "v12"
                else:
                    # 混合
                    total = v12_mae + hier_mae
                    if total > 0:
                        w_hier = v12_mae / total  # 误差越大权重越小
                        self.weights[group] = {"mix": True, "w_hier": w_hier}
                    else:
                        self.weights[group] = "hier"


# ===================== 9. 核心预测函数 =====================
def predict_single(
    t_train: np.ndarray,
    dE_train: np.ndarray,
    t_pred: float,
    sample: str,
    hmodel: HierarchicalModel,
    v12model: V12AutoOpt,
    shu_red_model: ShuRedModel,
    weight_config: dict,
    df_train: pd.DataFrame,
    cond: str = "UV",
) -> float:
    group = detect_group(sample)

    # 曙红组用专用模型
    if weight_config.get(group) == "shu_red_special":
        return shu_red_model.predict(t_train, dE_train, t_pred, sample, df_train)

    p_v12 = v12model.predict(t_train, dE_train, t_pred, group)
    p_hier = hmodel.predict(group, t_train, dE_train, t_pred)

    if p_hier is None and p_v12 is None:
        return 0.0
    if p_hier is None:
        return p_v12
    if p_v12 is None:
        return p_hier

    wc = weight_config.get(group, "v12")
    if isinstance(wc, dict) and wc.get("mix"):
        w_hier = wc["w_hier"]
        return float((1 - w_hier) * p_v12 + w_hier * p_hier)
    elif wc == "hier":
        return p_hier
    else:
        return p_v12


# ===================== 10. 测试集预测 =====================
def predict_testset(df_train, df_test, hmodel, v12model, shu_red_model, weight_config):
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

        pred = predict_single(t_arr, dE_arr, t_pred, sample, hmodel, v12model,
                              shu_red_model, weight_config, df_train, cond)
        results.append(pred)

    return np.array(results, dtype=float)


# ===================== 11. 前向外推评估 =====================
def forward_eval(df_train, hmodel, v12model, shu_red_model, weight_config):
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
            pred = predict_single(t_arr, dE_arr, t_tgt, sample, hmodel, v12model,
                                  shu_red_model, weight_config, df_train, cond)

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
    print("  颜料老化色差预测 v24")
    print("  v13底座 + 曙红专用模型 + 参数自动优化 + 远距离保护")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # Step 1: 构建分层模型
    print("\n[模型] 构建分层模型...")
    hmodel = HierarchicalModel(df_train)
    for g, m in hmodel.models.items():
        print(f"  {g:15s}: n={m['n']:.3f}, A={m['A']:.4f}, 成员={len(m['members'])}")

    # Step 2: 自动优化v12参数
    print("\n[优化] 网格搜索v12参数(ws,n)...")
    v12model = V12AutoOpt(df_train)
    for g, p in v12model.group_params.items():
        print(f"  {g:15s}: ws={p['ws']:.2f}, n={p['n']:.2f}, CV_MAE={p['cv_mae']:.4f}")

    # Step 3: 构建曙红专用模型
    print("\n[模型] 构建曙红专用模型...")
    shu_red_model = ShuRedModel(df_train)
    print(f"  组中位数dE: {shu_red_model.group_median_dE}")
    print(f"  组增长率: {shu_red_model.group_growth_rate:.4f}")

    # Step 4: 优化混合权重
    print("\n[优化] 搜索最优混合权重...")
    wopt = WeightOptimizer(df_train, hmodel, v12model, shu_red_model)
    for g, w in wopt.weights.items():
        print(f"  {g:15s}: {w}")

    # 前向验证
    print("\n[评估] 前向外推评估...")
    metrics = forward_eval(df_train, hmodel, v12model, shu_red_model, wopt.weights)
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
    test_preds = predict_testset(df_train, df_test, hmodel, v12model, shu_red_model, wopt.weights)
    print(f"  预测范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  预测均值: {test_preds.mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: test_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d)"
              f"  →  {test_preds[i]:.4f}")

    # 对比v13预测
    print("\n[对比] v24 vs v13:")
    print(f"  v24: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
    print(f"  v13: R²=0.9784, MAE=0.2906 (submitted score: 61.9)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
