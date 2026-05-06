"""
改进版 v6：颜料老化色差预测
- 去掉随机森林（RF太弱拖累融合）
- 只用3个强模型：XGBoost + LightGBM + CatBoost
- Optuna自动调参（LightGBM调参）
- 多seed融合（5个不同随机种子）
- log目标变换
- 目标编码
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV = BASE_DIR / "baseline_and_data" / "paint_aging_testset.csv"
OUT_CSV = BASE_DIR / "5" / "predict_out.csv"

TARGET = "dietaE"

# ===================== 特征工程 =====================
def _paint_category(name):
    if "染料" in name: return "natural_dye"
    if "皮纸" in name: return "base_paper"
    if "中国画" in name: return "chinese_paint"
    if "颜彩" in name: return "japanese_paint"
    if "矿物颜料" in name: return "mineral_paint"
    return "other"

def _paint_subcategory(name):
    if "染料" in name: return "dye"
    if "皮纸" in name: return "paper"
    if "孔雀蓝" in name: return "peacock_blue"
    if "大红" in name: return "red"
    if "柠檬黄" in name: return "lemon_yellow"
    if "曙红" in name: return "shu_red"
    if "翡翠绿" in name: return "jade_green"
    if "钴蓝" in name: return "cobalt_blue"
    if "浅葱" in name: return "asagi"
    if "鲜光黄" in name: return "bright_yellow"
    if "天蓝" in name: return "sky_blue"
    if "明黄" in name: return "bright_yello2"
    return "other"

def add_features(df, target_encoding_map=None, fit=False):
    df = df.copy()
    df["paint_category"] = df["sample"].apply(_paint_category)
    df["paint_subcategory"] = df["sample"].apply(_paint_subcategory)
    df["cond_uv"] = (df["aging_condition"] == "UV").astype(float)

    sat = np.sqrt(df["a0"]**2 + df["b0"]**2)
    df["color_saturation"] = sat
    df["hue_angle"] = np.degrees(np.arctan2(df["b0"], df["a0"]))
    df["time_log"] = np.log1p(df["aging_time_day"])
    df["time_sqrt"] = np.sqrt(df["aging_time_day"])
    df["time_sq"] = df["aging_time_day"] ** 2
    df["a0_b0_ratio"] = df["a0"] / (df["b0"].abs() + 1e-6)
    df["L0_sat_ratio"] = df["L0"] / (sat + 1e-6)
    df["time_x_sat"] = df["aging_time_day"] * sat
    df["time_x_L0"] = df["aging_time_day"] * df["L0"]
    df["time_x_a0"] = df["aging_time_day"] * df["a0"]
    df["time_x_b0"] = df["aging_time_day"] * df["b0"]
    df["time_log_x_sat"] = df["time_log"] * sat
    df["cond_x_time"] = df["cond_uv"] * df["aging_time_day"]
    df["cond_x_sat"] = df["cond_uv"] * sat
    df["time_x_sat_x_cond"] = df["time_x_sat"] * df["cond_uv"]
    df["L0_sq"] = df["L0"] ** 2
    df["a0_sq"] = df["a0"] ** 2
    df["b0_sq"] = df["b0"] ** 2
    df["time_cube"] = df["aging_time_day"] ** 3
    df["sat_x_L0"] = sat * df["L0"]

    if fit and TARGET in df.columns:
        te_cat = df.groupby("paint_category")[TARGET].mean()
        te_sub = df.groupby("paint_subcategory")[TARGET].mean()
        target_encoding_map = {"cat": te_cat.to_dict(), "sub": te_sub.to_dict()}
        df["te_category"] = df["paint_category"].map(te_cat)
        df["te_subcategory"] = df["paint_subcategory"].map(te_sub)
    elif target_encoding_map is not None:
        df["te_category"] = df["paint_category"].map(target_encoding_map["cat"])
        df["te_subcategory"] = df["paint_subcategory"].map(target_encoding_map["sub"])
    else:
        df["te_category"] = 0
        df["te_subcategory"] = 0

    return df, target_encoding_map

NUM_FEATURES = [
    "aging_time_day", "L0", "a0", "b0",
    "color_saturation", "hue_angle",
    "time_log", "time_sqrt", "time_sq",
    "a0_b0_ratio", "L0_sat_ratio",
    "time_x_sat", "time_x_L0", "time_x_a0", "time_x_b0",
    "time_log_x_sat", "cond_x_time", "cond_x_sat",
    "time_x_sat_x_cond",
    "L0_sq", "a0_sq", "b0_sq",
    "time_cube", "sat_x_L0",
    "te_category", "te_subcategory",
]
CAT_FEATURES = ["paint_category"]

def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUM_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), CAT_FEATURES),
    ])

# ===================== Optuna调参 =====================
def optuna_objective(trial, X, y_log, groups):
    """Optuna目标函数（用LightGBM调参）"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }

    model = LGBMRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    gkf = GroupKFold(n_splits=5)
    scores = []
    for tr_idx, va_idx in gkf.split(X, y_log, groups=groups):
        prep = build_preprocessor()
        pipe = Pipeline([("pre", prep), ("m", model)])
        pipe.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
        pred = pipe.predict(X.iloc[va_idx])
        pred_orig = np.expm1(pred)
        true_orig = np.expm1(y_log.iloc[va_idx])
        scores.append(r2_score(true_orig, pred_orig))

    return np.mean(scores)

# ===================== 主程序 =====================
def main():
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"[数据] 训练集 {len(df_train)} 行, 测试集 {len(df_test)} 行")

    # 特征工程
    df_train_fe, te_map = add_features(df_train, fit=True)
    df_test_fe, _ = add_features(df_test, target_encoding_map=te_map)

    X = df_train_fe[NUM_FEATURES + CAT_FEATURES]
    y = df_train_fe[TARGET]
    X_test = df_test_fe[NUM_FEATURES + CAT_FEATURES]

    # log变换
    y_log = np.log1p(y)
    groups = df_train_fe["sample"].values
    gkf = GroupKFold(n_splits=5)
    seeds = [42, 123, 456, 789, 2024]

    # ===== Step 1: Optuna自动调参 =====
    print("\n========== Step 1: Optuna自动调参 ==========")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X, y_log, groups),
        n_trials=80,
        show_progress_bar=False
    )
    best_params = study.best_params
    best_score = study.best_value
    print(f"  Optuna最佳R²: {best_score:.4f}")
    print(f"  最佳参数: {best_params}")

    # ===== Step 2: 用最佳参数构建3个模型 =====
    print("\n========== Step 2: 多seed交叉验证 (3模型 x 5seed) ==========")

    def make_xgb(seed):
        return XGBRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            min_child_weight=best_params["min_child_weight"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            random_state=seed, n_jobs=-1, verbosity=0
        )

    def make_lgbm(seed):
        return LGBMRegressor(
            **best_params,
            random_state=seed, n_jobs=-1, verbose=-1
        )

    def make_cat(seed):
        return CatBoostRegressor(
            iterations=best_params["n_estimators"],
            depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bylevel=best_params["colsample_bytree"],
            l2_leaf_reg=max(best_params["reg_lambda"], 1),
            random_state=seed,
            verbose=0, thread_count=-1
        )

    base_classes = [
        ("xgb", make_xgb),
        ("lgbm", make_lgbm),
        ("cat", make_cat),
    ]

    # 收集所有OOF和test预测（3模型 x 5seed = 15个）
    all_oof = []
    all_test = []

    for si, seed in enumerate(seeds):
        print(f"\n  --- Seed {seed} ({si+1}/{len(seeds)}) ---")
        for name, make_fn in base_classes:
            model = make_fn(seed)
            oof_pred = np.zeros(len(X))
            test_pred = np.zeros(len(X_test))

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
                prep = build_preprocessor()
                pipe = Pipeline([("pre", prep), ("m", model)])
                pipe.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
                oof_pred[va_idx] = np.expm1(pipe.predict(X.iloc[va_idx]))
                test_pred += np.expm1(pipe.predict(X_test)) / 5.0

            all_oof.append(oof_pred)
            all_test.append(test_pred)

            mae = mean_absolute_error(y, oof_pred)
            r2 = r2_score(y, oof_pred)
            print(f"    {name}: MAE={mae:.4f}, R2={r2:.4f}")

    # ===== Step 3: 全量训练最终预测 =====
    print("\n========== Step 3: 融合对比 ==========")
    oof_stack = np.column_stack(all_oof)
    test_stack = np.column_stack(all_test)

    # 简单平均
    oof_avg = np.mean(oof_stack, axis=1)
    test_avg = np.mean(test_stack, axis=1)
    mae_avg = mean_absolute_error(y, oof_avg)
    rmse_avg = np.sqrt(mean_squared_error(y, oof_avg))
    r2_avg = r2_score(y, oof_avg)
    print(f"  简单平均: MAE={mae_avg:.4f}, RMSE={rmse_avg:.4f}, R2={r2_avg:.4f}")

    # Ridge融合
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_stack, y)
    oof_ridge = ridge.predict(oof_stack)
    test_ridge = ridge.predict(test_stack)
    mae_ridge = mean_absolute_error(y, oof_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y, oof_ridge))
    r2_ridge = r2_score(y, oof_ridge)
    print(f"  Ridge融合: MAE={mae_ridge:.4f}, RMSE={rmse_ridge:.4f}, R2={r2_ridge:.4f}")

    # CatBoost单独平均（5个seed）
    cat_oof = np.mean([all_oof[i] for i in range(2, len(all_oof), 3)], axis=0)
    cat_test = np.mean([all_test[i] for i in range(2, len(all_test), 3)], axis=0)
    mae_cat = mean_absolute_error(y, cat_oof)
    rmse_cat = np.sqrt(mean_squared_error(y, cat_oof))
    r2_cat = r2_score(y, cat_oof)
    print(f"  CatBoost单独: MAE={mae_cat:.4f}, RMSE={rmse_cat:.4f}, R2={r2_cat:.4f}")

    # 选最好的
    results = [
        ("简单平均", test_avg, mae_avg, rmse_avg, r2_avg),
        ("Ridge融合", test_ridge, mae_ridge, rmse_ridge, r2_ridge),
        ("CatBoost单独", cat_test, mae_cat, rmse_cat, r2_cat),
    ]
    best = max(results, key=lambda x: x[4])
    final_pred = best[1]
    best_method = best[0]
    best_mae = best[2]
    best_rmse = best[3]
    best_r2 = best[4]

    final_pred = np.maximum(final_pred, 0)

    # 保存
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET: final_pred}).to_csv(OUT_CSV, index=False)

    print(f"\n========== 最终结果 ==========")
    print(f"  最佳方法: {best_method}")
    print(f"  MAE: {best_mae:.4f}")
    print(f"  RMSE: {best_rmse:.4f}")
    print(f"  R2: {best_r2:.4f}")
    print(f"  预测范围: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    print(f"  已保存: {OUT_CSV}")

if __name__ == "__main__":
    main()
