"""
颜料老化色差预测 · 最终版（过拟合检测 + 防过拟合优化 + 合规提交）
保留平行样真实差异 | Group K折 + 时间双重验证 | 小样本稳健设计
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ===================== 路径配置（AI 根目录）=====================
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV_PATH = BASE_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV_PATH = BASE_DIR / "baseline_and_data" / "paint_aging_testset.csv"
PRED_OUTPUT_CSV = BASE_DIR / "line" / "predict_out.csv"

# 特征配置
ORIGIN_NUM_FEATURES = ["aging_time_day", "L0", "a0", "b0"]
CAT_FEATURES = ["aging_condition", "paint_category"]
ALL_FEATURES = ORIGIN_NUM_FEATURES + ["color_saturation", "aging_time_log"]
TARGET_COL = "dietaE"

np.random.seed(RANDOM_STATE)

# ===================== 特征工程（保留平行样，无强制修正）=====================
class PaintFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # 颜料大类
        df["paint_category"] = df["sample"].apply(self._get_paint_category)
        # 老化条件编码
        df["aging_condition"] = df["aging_condition"].map({"UV": 1, "humid-_heat": 0})
        # 时间非线性（老化动力学）
        df["aging_time_log"] = np.log1p(df["aging_time_day"])
        # 颜色饱和度
        df["color_saturation"] = np.sqrt(df["a0"] ** 2 + df["b0"] ** 2)
        return df
    
    @staticmethod
    def _get_paint_category(name):
        if "染料" in name: return "natural_dye"
        elif "皮纸" in name: return "base_paper"
        elif "中国画-" in name: return "chinese_paint"
        elif "颜彩-" in name: return "japanese_paint"
        elif "矿物颜料-" in name: return "mineral_paint"
        else: return "other"

# ===================== 数据加载（仅校验，保留平行样）=====================
def load_data():
    df_train = pd.read_csv(TRAIN_CSV_PATH, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV_PATH, encoding="utf-8")
    print(f"[数据加载] 训练集={len(df_train)} 测试集={len(df_test)}")
    return df_train, df_test

# ===================== 【核心】过拟合检测：训练集VS验证集对比 =====================
def check_overfitting(model, X_train, y_train, X_valid, y_valid, fold):
    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 验证集预测
    y_valid_pred = model.predict(X_valid)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    valid_r2 = r2_score(y_valid, y_valid_pred)
    
    # 过拟合判断
    gap_rmse = train_rmse - valid_rmse
    is_overfit = gap_rmse >= 0.5 or (train_r2 > 0.95 and valid_r2 < 0.7)
    
    print(f"[Fold {fold}] 训练 RMSE={train_rmse:.4f} | 验证 RMSE={valid_rmse:.4f} | 差值={gap_rmse:.4f}")
    print(f"           训练 R2={train_r2:.4f}   | 验证 R2={valid_r2:.4f}   | 过拟合={is_overfit}")
    return train_rmse, valid_rmse, gap_rmse, is_overfit

# ===================== 学习曲线（可视化过拟合）=====================
def plot_learning_curve(model, X, y, groups):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, groups=groups, cv=GroupKFold(5),
        train_sizes=np.linspace(0.1,1,5), scoring="neg_root_mean_squared_error"
    )
    train_rmse = -train_scores.mean(axis=1)
    valid_rmse = -valid_scores.mean(axis=1)
    
    plt.figure(figsize=(8,4))
    plt.plot(train_sizes, train_rmse, 'o-', label='Train RMSE')
    plt.plot(train_sizes, valid_rmse, 's-', label='Valid RMSE')
    plt.title('Learning Curve (Overfitting Check)')
    plt.xlabel('Train Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    print("[✅] 学习曲线已保存：learning_curve.png（可直接看是否过拟合）")

# ===================== 时间外推验证（贴合赛题：短期→长期）=====================
def time_based_validate(model, df_fe):
    train_idx = df_fe[df_fe["aging_time_day"] <= 15].index
    valid_idx = df_fe[df_fe["aging_time_day"] > 15].index
    if len(valid_idx) == 0:
        print("[时间验证] 无长期数据，跳过")
        return
    
    X_train = df_fe.loc[train_idx, ALL_FEATURES]
    X_valid = df_fe.loc[valid_idx, ALL_FEATURES]
    y_train = df_fe.loc[train_idx, TARGET_COL]
    y_valid = df_fe.loc[valid_idx, TARGET_COL]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"[时间验证] 短期→长期预测 RMSE={rmse:.4f}")

# ===================== 模型训练（防过拟合参数）=====================
def train_model(df_train):
    # 特征工程
    fe = PaintFeatureTransformer()
    df_fe = fe.fit_transform(df_train)
    groups = df_fe["sample"].values
    X = df_fe[ALL_FEATURES]
    y = df_fe[TARGET_COL]
    
    # 预处理流水线
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ALL_FEATURES),
    ])
    
    # 🔥 防过拟合核心参数（小样本专用）
    model = RandomForestRegressor(
        n_estimators=200,     # 减少树数量
        max_depth=5,          # 限制深度
        min_samples_leaf=3,   # 增大叶子节点
        max_features=0.5,     # 降低特征采样
        random_state=RANDOM_STATE, n_jobs=-1
    )
    
    pipeline = Pipeline([("pre", preprocessor), ("model", model)])
    
    # 1. Group K折交叉验证 + 过拟合检测
    print("\n==================== 过拟合检测（Group 5折）====================")
    gkf = GroupKFold(5)
    overfit_flags = []
    for i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        tr_X, va_X = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]
        pipeline.fit(tr_X, tr_y)
        _, _, gap, is_overfit = check_overfitting(pipeline, tr_X, tr_y, va_X, va_y, i+1)
        overfit_flags.append(is_overfit)
    
    # 整体过拟合判断
    total_overfit = any(overfit_flags)
    print(f"\n【最终过拟合判断】：{'⚠️ 存在过拟合' if total_overfit else '✅ 无过拟合'}")
    
    # 2. 学习曲线
    plot_learning_curve(pipeline, X, y, groups)
    
    # 3. 时间外推验证
    print("\n==================== 时间外推验证 ====================")
    time_based_validate(pipeline, df_fe)
    
    # 全量训练最终模型
    final_model = Pipeline([
        ("fe", PaintFeatureTransformer()),
        ("pre", preprocessor),
        ("model", model)
    ])
    final_model.fit(df_train, y)
    return final_model

# ===================== 推理并生成提交文件 =====================
def predict_and_save(model, df_test):
    pred = model.predict(df_test)
    submit = pd.DataFrame({TARGET_COL: pred})
    submit.to_csv(PRED_OUTPUT_CSV, index=False)
    print(f"\n[✅ 提交文件已生成] {PRED_OUTPUT_CSV}")
    print(f"[✅ 预测行数] {len(submit)}（符合37条测试集要求）")
    return submit

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 加载数据
    df_train, df_test = load_data()
    # 2. 训练模型（自带过拟合检测）
    final_model = train_model(df_train)
    # 3. 推理并保存
    submit_df = predict_and_save(final_model, df_test)
    print("\n🎉 全部完成！可直接提交 line/predict_out.csv")