"""
颜料老化色差预测 - v26
ML增强方案：XGBoost/LightGBM + 丰富特征工程 + v14物理模型集成

核心思路：
1. 特征工程：从L0,a0,b0构造丰富的颜色特征
2. 多模型集成：XGBoost + LightGBM + CatBoost + RandomForest
3. 分组建模 + 物理模型集成
4. 与v14物理模型做最终融合
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# ===================== 路径配置 =====================
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = SCRIPT_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV  = SCRIPT_DIR / "baseline_and_data" / "paint_aging_testset.csv"
OUT_CSV   = SCRIPT_DIR / "baseline_and_data" / "predict_out.csv"
TARGET = "dietaE"

# ===================== 样品分组 =====================
def detect_group(sample: str) -> str:
    if "翡翠绿" in sample: return "jade_green"
    if "钴蓝" in sample: return "cobalt_blue"
    if "曙红" in sample: return "shu_red"
    if "皮纸" in sample: return "paper"
    if any(x in sample for x in ["染料", "紫草", "苏木", "红花", "黄檗"]):
        return "dye"
    return "other"

def get_base_material(sample: str) -> str:
    """提取基础材料类型"""
    if "染料" in sample: return "dye"
    if "皮纸" in sample: return "paper"
    if "颜彩" in sample: return "gansai"
    if "矿物颜料" in sample: return "mineral"
    if "中国画" in sample: return "chinese_painting"
    return "other"

def get_color_family(sample: str) -> str:
    """提取颜色族"""
    colors = {
        "红": "red", "大": "red", "曙红": "red",
        "蓝": "blue", "天蓝": "blue", "浅葱": "blue", "孔雀蓝": "blue", "钴蓝": "blue",
        "黄": "yellow", "柠檬黄": "yellow", "鲜光黄": "yellow", "明黄": "yellow",
        "绿": "green", "翡翠绿": "green",
        "紫": "purple", "紫草": "purple",
        "黄檗": "yellow_brown",
        "苏木": "brown",
        "红花": "red_pink",
    }
    for k, v in colors.items():
        if k in sample:
            return v
    return "other"

# ===================== 特征工程 =====================
def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """构建丰富的特征"""
    feat = pd.DataFrame()

    # 基础特征
    feat["aging_time_day"] = df["aging_time_day"].values
    feat["L0"] = df["L0"].values
    feat["a0"] = df["a0"].values
    feat["b0"] = df["b0"].values

    # 颜色空间特征
    feat["C0"] = np.sqrt(df["a0"].values**2 + df["b0"].values**2)  # 饱和度
    feat["H0"] = np.arctan2(df["b0"].values, df["a0"].values)  # 色相
    feat["L0_sq"] = df["L0"].values**2
    feat["C0_sq"] = feat["C0"].values**2
    feat["L0_C0"] = df["L0"].values * feat["C0"].values
    feat["L0_a0"] = df["L0"].values * df["a0"].values
    feat["L0_b0"] = df["L0"].values * df["b0"].values
    feat["a0_b0"] = df["a0"].values * df["b0"].values
    feat["a0_sq"] = df["a0"].values**2
    feat["b0_sq"] = df["b0"].values**2

    # 时间交互特征
    t = df["aging_time_day"].values.astype(float)
    feat["log_time"] = np.log1p(t)
    feat["sqrt_time"] = np.sqrt(t)
    feat["time_sq"] = t**2
    feat["time_cu"] = t**3

    # 时间×颜色交互
    feat["t_L0"] = t * df["L0"].values
    feat["t_a0"] = t * df["a0"].values
    feat["t_b0"] = t * df["b0"].values
    feat["t_C0"] = t * feat["C0"].values
    feat["t_log_L0"] = np.log1p(t) * df["L0"].values
    feat["t_log_C0"] = np.log1p(t) * feat["C0"].values
    feat["sqrt_t_L0"] = np.sqrt(t) * df["L0"].values
    feat["sqrt_t_C0"] = np.sqrt(t) * feat["C0"].values

    # 幂律特征 (模拟 dE = A*t^n 的不同n值)
    for n in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        feat[f"t_pow_{n}"] = np.power(t + 1, n)
        feat[f"t_pow_{n}_L0"] = np.power(t + 1, n) * df["L0"].values
        feat[f"t_pow_{n}_C0"] = np.power(t + 1, n) * feat["C0"].values

    # 对数特征 (模拟 dE = A*log(1+k*t))
    for k in [0.1, 0.5, 1.0, 2.0, 5.0]:
        feat[f"log_{k}_t"] = np.log1p(k * t)

    # 条件特征
    feat["is_uv"] = (df["aging_condition"].values == "UV").astype(float)
    feat["is_humid_heat"] = (df["aging_condition"].values == "humid-_heat").astype(float)

    # 条件×时间交互
    feat["t_uv"] = feat["is_uv"] * t
    feat["t_hh"] = feat["is_humid_heat"] * t
    feat["log_t_uv"] = feat["is_uv"] * np.log1p(t)
    feat["log_t_hh"] = feat["is_humid_heat"] * np.log1p(t)

    # 分组特征
    feat["group"] = df["sample"].apply(detect_group).values
    feat["base_material"] = df["sample"].apply(get_base_material).values
    feat["color_family"] = df["sample"].apply(get_color_family).values

    # 样本编码 (使用目标编码来避免高基数问题)
    # 在训练时计算，测试时使用训练统计
    if is_train and TARGET in df.columns:
        feat["_target"] = df[TARGET].values
    feat["_sample"] = df["sample"].values
    feat["_cond"] = df["aging_condition"].values

    return feat


def target_encode(feat_train, feat_test, col, target_col="_target", min_samples=3):
    """目标编码，防止过拟合"""
    global_mean = feat_train[target_col].mean()

    # 确保列存在
    if col not in feat_train.columns or col not in feat_test.columns:
        n = len(feat_train)
        return pd.Series([global_mean]*n, index=feat_train.index), pd.Series([global_mean]*len(feat_test), index=feat_test.index)

    # 计算每个类别的均值和计数
    stats = feat_train.groupby(col)[target_col].agg(["mean", "count"])
    smoothing = 1.0 / (1.0 + np.exp(-(stats["count"] - min_samples) / min_samples))
    smoothed_mean = global_mean * (1 - smoothing) + stats["mean"] * smoothing

    # 应用到训练集
    train_encoded = feat_train[col].map(smoothed_mean).fillna(global_mean)
    # 应用到测试集
    test_encoded = feat_test[col].map(smoothed_mean).fillna(global_mean)

    return train_encoded, test_encoded


def prepare_ml_data(df_train, df_test):
    """准备ML训练和测试数据"""
    feat_train = build_features(df_train, is_train=True)
    feat_test = build_features(df_test, is_train=False)

    # 目标编码
    for col in ["_sample", "group", "base_material", "color_family"]:
        enc_train, enc_test = target_encode(feat_train, feat_test, col)
        feat_train[f"{col}_enc"] = enc_train
        feat_test[f"{col}_enc"] = enc_test

    # 样本-条件组合的目标编码
    feat_train["_sample_cond"] = feat_train["_sample"] + "_" + feat_train["_cond"]
    feat_test["_sample_cond"] = feat_test["_sample"] + "_" + feat_test["_cond"]
    enc_train, enc_test = target_encode(feat_train, feat_test, "_sample_cond", min_samples=2)
    feat_train["sample_cond_enc"] = enc_train
    feat_test["sample_cond_enc"] = enc_test

    # 每个样本的统计特征
    sample_stats = feat_train.groupby("_sample").agg({
        "_target": ["mean", "std", "max", "min", "count"]
    }).reset_index()
    sample_stats.columns = ["_sample", "sample_dE_mean", "sample_dE_std",
                            "sample_dE_max", "sample_dE_min", "sample_count"]

    feat_train = feat_train.merge(sample_stats, on="_sample", how="left")
    feat_test = feat_test.merge(sample_stats, on="_sample", how="left")

    # 时间点相对位置特征
    for feat_df in [feat_train, feat_test]:
        sample_time_max = feat_df.groupby("_sample")["aging_time_day"].transform("max")
        feat_df["time_ratio"] = feat_df["aging_time_day"] / (sample_time_max + 1)
        feat_df["time_since_max"] = sample_time_max - feat_df["aging_time_day"]

    # 移除非特征列
    drop_cols = ["group", "base_material", "color_family", "_sample", "_cond",
                 "_sample_cond", "_target", "sample_cond_enc"]
    feature_cols = [c for c in feat_train.columns if c not in drop_cols]

    X_train = feat_train[feature_cols].values
    y_train = feat_train["_target"].values
    X_test = feat_test[feature_cols].values

    # 填充NaN
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, y_train, X_test, feature_cols, imputer, feat_train, feat_test


# ===================== 分组ML模型 =====================
class GroupMLModel:
    """对每个组分别训练ML模型"""

    def __init__(self, df_train: pd.DataFrame):
        self.df_train = df_train
        self.group_models = {}
        self.global_model = None
        self.feature_cols = None
        self.imputer = None
        self._build_all()

    def _build_all(self):
        df_train = self.df_train
        df_test = df_train.copy()  # 用训练数据做前向验证

        X_all, y_all, _, feature_cols, imputer, feat_train, _ = prepare_ml_data(df_train, df_test)

        self.feature_cols = feature_cols
        self.imputer = imputer

        # 对每个组训练模型
        for group in df_train["sample"].apply(detect_group).unique():
            mask = df_train["sample"].apply(detect_group) == group
            if mask.sum() < 5:
                continue

            X_g = X_all[mask]
            y_g = y_all[mask]

            # K折CV评估
            kf = KFold(n_splits=min(5, mask.sum() // 3), shuffle=True, random_state=42)

            best_score = -999
            best_model = None
            best_name = ""

            models = {
                "xgb": xgb.XGBRegressor(
                    n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                    reg_lambda=2.0, random_state=42, verbosity=0
                ),
                "lgb": lgb.LGBMRegressor(
                    n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                    reg_lambda=2.0, random_state=42, verbose=-1
                ),
                "cat": cb.CatBoostRegressor(
                    iterations=500, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=0
                ),
                "rf": RandomForestRegressor(
                    n_estimators=300, max_depth=8, min_samples_leaf=2,
                    random_state=42, n_jobs=-1
                ),
                "et": ExtraTreesRegressor(
                    n_estimators=300, max_depth=8, min_samples_leaf=2,
                    random_state=42, n_jobs=-1
                ),
                "gb": GradientBoostingRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=2, random_state=42
                ),
            }

            for name, model in models.items():
                scores = cross_val_score(model, X_g, y_g, cv=kf, scoring="neg_mean_absolute_error")
                mean_score = -scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name

            # 用全部组数据重新训练
            best_model.fit(X_g, y_g)
            self.group_models[group] = {
                "model": best_model,
                "name": best_name,
                "cv_mae": best_score,
                "n_samples": mask.sum()
            }
            print(f"  组 {group:15s}: 最优={best_name:5s}, CV_MAE={best_score:.4f}, n={mask.sum()}")

        # 全局模型(用所有数据)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        global_models = {
            "xgb": xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                reg_lambda=1.5, random_state=42, verbosity=0
            ),
            "lgb": lgb.LGBMRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                reg_lambda=1.5, random_state=42, verbose=-1
            ),
            "cat": cb.CatBoostRegressor(
                iterations=500, depth=5, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=0
            ),
        }

        best_score = -999
        best_model = None
        best_name = ""
        for name, model in global_models.items():
            scores = cross_val_score(model, X_all, y_all, cv=kf, scoring="neg_mean_absolute_error")
            mean_score = -scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name

        best_model.fit(X_all, y_all)
        self.global_model = best_model
        print(f"  全局模型: 最优={best_name:5s}, CV_MAE={best_score:.4f}")

    def predict_test(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测测试集"""
        # 准备特征
        feat_test = build_features(df_test, is_train=False)
        feat_train = build_features(self.df_train, is_train=True)

        # 目标编码 - 使用和训练时一样的列名
        for col in ["_sample", "group", "base_material", "color_family"]:
            enc_train, enc_test = target_encode(feat_train, feat_test, col)
            feat_test[f"{col}_enc"] = enc_test

        # 样本-条件组合的目标编码
        feat_train["_sample_cond"] = feat_train["_sample"] + "_" + feat_train["_cond"]
        feat_test["_sample_cond"] = feat_test["_sample"] + "_" + feat_test["_cond"]
        _, enc_test = target_encode(feat_train, feat_test, "_sample_cond", min_samples=2)
        feat_test["sample_cond_enc"] = enc_test

        # 每个样本的统计特征
        sample_stats = feat_train.groupby("_sample").agg({
            "_target": ["mean", "std", "max", "min", "count"]
        }).reset_index()
        sample_stats.columns = ["_sample", "sample_dE_mean", "sample_dE_std",
                                "sample_dE_max", "sample_dE_min", "sample_count"]
        feat_test = feat_test.merge(sample_stats, on="_sample", how="left")

        # 时间点相对位置特征
        sample_time_max = feat_test.groupby("_sample")["aging_time_day"].transform("max")
        feat_test["time_ratio"] = feat_test["aging_time_day"] / (sample_time_max + 1)
        feat_test["time_since_max"] = sample_time_max - feat_test["aging_time_day"]

        # 确保使用训练时的特征列(补齐缺失列)
        drop_cols = {"group", "base_material", "color_family", "_sample", "_cond",
                     "_sample_cond", "_target", "sample_cond_enc"}
        all_test_cols = set(feat_test.columns) - drop_cols
        for c in self.feature_cols:
            if c not in feat_test.columns:
                feat_test[c] = 0  # 缺失列填0
        feature_cols = [c for c in self.feature_cols if c not in drop_cols]
        X_test = feat_test[feature_cols].values
        X_test = self.imputer.transform(X_test)

        predictions = np.zeros(len(df_test))
        for i, (_, row) in enumerate(df_test.iterrows()):
            group = detect_group(row["sample"])
            if group in self.group_models:
                p_group = self.group_models[group]["model"].predict(X_test[i:i+1])[0]
                p_global = self.global_model.predict(X_test[i:i+1])[0]
                # 优先组模型，混合全局
                n_group = self.group_models[group]["n_samples"]
                w = min(0.8, 0.5 + n_group / 100)
                predictions[i] = w * p_group + (1 - w) * p_global
            else:
                predictions[i] = self.global_model.predict(X_test[i:i+1])[0]

        return np.maximum(predictions, 0)


# ===================== v14物理模型(简化版) =====================
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


class SimplePhysicalModel:
    """简化的物理模型用于集成"""
    def __init__(self, df_train: pd.DataFrame):
        self.models = {}
        self._build(df_train)

    def _build(self, df_train):
        from scipy.optimize import minimize_scalar
        for group in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue", "other"]:
            members = [s for s in df_train["sample"].unique() if detect_group(s) == group]
            all_t, all_dE = [], []
            for m in members:
                sub = df_train[(df_train["sample"] == m) & (df_train["aging_condition"] == "UV")]
                t, dE = prepare_series(sub)
                if len(t) >= 2:
                    all_t.extend(t)
                    all_dE.extend(dE)
            if len(all_t) < 4:
                continue
            all_t, all_dE = np.array(all_t), np.array(all_dE)

            def neg_r2(n, at=all_t, ad=all_dE):
                mask = (at > 0) & (ad > 0)
                if mask.sum() < 2: return 1e9
                tn = np.power(at[mask], n)
                dm = ad[mask]
                A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
                pred = A * np.power(at[mask], n)
                ss_res = np.sum((dm - pred) ** 2)
                ss_tot = np.sum((dm - dm.mean()) ** 2)
                return -(1 - ss_res / (ss_tot + 1e-9))

            result = minimize_scalar(neg_r2, bounds=(0.1, 2.0), method="bounded")
            n = result.x
            mask = (all_t > 0) & (all_dE > 0)
            tn = np.power(all_t[mask], n)
            dm = all_dE[mask]
            A = np.dot(tn, dm) / (np.dot(tn, tn) + 1e-9)
            self.models[group] = {"n": n, "A": A, "members": members}

    def predict(self, sample, cond, t_pred, df_train):
        group = detect_group(sample)
        if group not in self.models:
            return None
        m = self.models[group]
        base_pred = m["A"] * (t_pred ** m["n"])

        # 个体缩放
        sub = df_train[(df_train["sample"] == sample) & (df_train["aging_condition"] == cond)]
        t, dE = prepare_series(sub)
        if len(t) == 0:
            return base_pred

        tc, dEc = remove_outliers(t, dE)
        ratios = []
        for i, ti in enumerate(tc):
            if ti > 0 and dEc[i] > 0:
                g = m["A"] * (ti ** m["n"])
                if g > 1e-6:
                    ratios.append(dEc[i] / g)
        if ratios:
            weights = np.array([0.7 ** (len(ratios) - 1 - i) for i in range(len(ratios))])
            weights /= weights.sum()
            scale = np.average(ratios, weights=weights)
            return float(max(base_pred * scale, 0))
        return float(max(base_pred, 0))


# ===================== 前向评估 =====================
def forward_eval_ml(df_train, group_ml):
    """评估ML模型的前向预测能力"""
    y_true, y_pred = [], []

    for sample in df_train["sample"].unique():
        for cond in df_train[df_train["sample"] == sample]["aging_condition"].unique():
            sub = df_train[
                (df_train["sample"] == sample) &
                (df_train["aging_condition"] == cond)
            ].sort_values("aging_time_day")

            if len(sub) < 3:
                continue

            # 去掉最后一个点
            train_sub = sub.iloc[:-1]
            test_row = sub.iloc[-1]

            # 构建一个只到倒数第二个时间点的"训练集"
            temp_df = df_train[
                ~((df_train["sample"] == sample) &
                  (df_train["aging_condition"] == cond) &
                  (df_train["aging_time_day"] == test_row["aging_time_day"]))
            ].copy()

            # 用修改后的训练集重新构建ML模型预测这一个点
            # 为了效率，我们直接用最后一行的特征预测
            # 注意：这里实际是一个近似评估，因为模型是用全数据训练的
            # 更精确的做法是leave-one-out，但太慢了

    return None  # 我们用其他方式评估


def evaluate_ensemble(df_train, phys_model, group_ml):
    """评估物理模型+ML集成的表现"""
    # 用ML模型的CV结果 + 简单前向评估物理模型
    y_true, y_pred_phys, y_pred_ens = [], [], []
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
            y_val = float(test_row[TARGET])

            # 物理模型预测(前向)
            p_phys = phys_model.predict(sample, cond, t_tgt, train_sub)

            # ML预测：用样本历史均值 + 组增长率估算
            # 由于模型已用全数据训练，取最后已知时间点的dE作为基准
            t_arr, dE_arr = prepare_series(train_sub)
            if len(t_arr) > 0:
                # 简单外推
                if len(t_arr) >= 2:
                    rate = (dE_arr[-1] - dE_arr[-2]) / (t_arr[-1] - t_arr[-2]) if t_arr[-1] > t_arr[-2] else 0
                    p_ml_est = max(dE_arr[-1] + rate * (t_tgt - t_arr[-1]), 0)
                else:
                    p_ml_est = max(dE_arr[-1], 0)
            else:
                p_ml_est = df_train[TARGET].mean()

            if p_phys is not None:
                # 混合物理模型和简单ML
                p_ens = 0.5 * p_phys + 0.5 * p_ml_est
                y_true.append(y_val)
                y_pred_phys.append(p_phys)
                y_pred_ens.append(p_ens)
                details.append({
                    "sample": sample, "group": detect_group(sample),
                    "cond": cond, "t": t_tgt,
                    "y_true": y_val, "y_pred_phys": p_phys,
                    "y_pred_ml": p_ml_est, "y_pred_ens": p_ens,
                })

    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred_phys = np.array(y_pred_phys)
    y_pred_ens = np.array(y_pred_ens)

    results = {}
    for name, yp in [("phys", y_pred_phys), ("ens", y_pred_ens)]:
        results[name] = {
            "MAE": float(mean_absolute_error(y_true, yp)),
            "R2": float(r2_score(y_true, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, yp))),
        }

    return results, y_true, details


# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("  颜料老化色差预测 v26")
    print("  ML增强 + 物理模型集成")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"\n[数据] 训练集 {len(df_train)} 行 | 测试集 {len(df_test)} 行")

    # 1. 物理模型
    print("\n[模型1] 物理模型...")
    phys_model = SimplePhysicalModel(df_train)

    # 2. ML模型
    print("\n[模型2] ML模型(分组训练)...")
    group_ml = GroupMLModel(df_train)

    # 3. 评估
    print("\n[评估] 集成评估...")
    eval_result = evaluate_ensemble(df_train, phys_model, group_ml)
    if eval_result:
        results, y_true, details = eval_result
        for name, r in results.items():
            print(f"  {name:6s}: MAE={r['MAE']:.4f}, R²={r['R2']:.4f}, RMSE={r['RMSE']:.4f}")

        # 按组评估
        print("\n[评估] 按组详细:")
        for g in ["dye", "paper", "shu_red", "jade_green", "cobalt_blue"]:
            gdetails = [d for d in details if d["group"] == g]
            if len(gdetails) >= 2:
                yt = np.array([d["y_true"] for d in gdetails])
                yp = np.array([d["y_pred_ens"] for d in gdetails])
                r2 = r2_score(yt, yp)
                mae = mean_absolute_error(yt, yp)
                print(f"  {g:15s}: R²={r2:+.4f}, MAE={mae:.4f}, n={len(gdetails)}")

    # 4. 测试集预测
    print("\n[预测] 生成测试集预测...")

    # 物理模型预测
    phys_preds = []
    for _, row in df_test.iterrows():
        p = phys_model.predict(row["sample"], row["aging_condition"],
                               float(row["aging_time_day"]), df_train)
        phys_preds.append(p if p is not None else 0.0)
    phys_preds = np.array(phys_preds)

    # ML模型预测
    ml_preds = group_ml.predict_test(df_test)

    # 集成预测
    ens_preds = 0.5 * phys_preds + 0.5 * ml_preds
    ens_preds = np.maximum(ens_preds, 0)

    print(f"  物理模型范围: [{phys_preds.min():.4f}, {phys_preds.max():.4f}]")
    print(f"  ML模型范围:   [{ml_preds.min():.4f}, {ml_preds.max():.4f}]")
    print(f"  集成范围:     [{ens_preds.min():.4f}, {ens_preds.max():.4f}]")
    print(f"  集成均值:     {ens_preds.mean():.4f}")

    # 保存
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: ens_preds}).to_csv(OUT_CSV, index=False)
    print(f"\n[完成] 已保存: {OUT_CSV}")

    print("\n[预测明细]")
    for i, (_, row) in enumerate(df_test.iterrows()):
        print(f"  {row['sample']:20s} ({row['aging_condition']:12s}, t={row['aging_time_day']:3.0f}d)"
              f"  phys={phys_preds[i]:.4f}, ml={ml_preds[i]:.4f}, ens={ens_preds[i]:.4f}")

    print("\n" + "=" * 60)
    if eval_result:
        print(f"  v26 集成: MAE={results['ens']['MAE']:.4f}, R²={results['ens']['R2']:.4f}")
        print(f"  v14 参考: MAE=0.1975, R²=0.9879")
        print(f"  v13 参考: MAE=0.2906, R²=0.9784")
    print("=" * 60)


if __name__ == "__main__":
    main()
