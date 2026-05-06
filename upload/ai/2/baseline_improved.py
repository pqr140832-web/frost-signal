"""
改进版Baseline：颜料老化色差预测模型
- 添加特征工程（颜色饱和度、对数变换、交互特征）
- 支持多模型集成（RandomForest + XGBoost）
- 改进的交叉验证和评估
- 详细的日志和错误处理
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ===================== 配置 =====================
RANDOM_STATE = 42
BASE_DIR = Path(__file__).parent.parent
TRAIN_CSV_PATH = BASE_DIR / "baseline_and_data" / "paint_aging_trainset.csv"
TEST_CSV_PATH = BASE_DIR / "baseline_and_data" / "paint_aging_testset.csv"
PRED_OUTPUT_CSV = BASE_DIR / "2" / "paint_pred.csv"

ORIGIN_NUM_FEATURES = ["aging_time_day", "L0", "a0", "b0"]
CAT_FEATURES = ["aging_condition", "paint_category"]
TARGET_COL = "dietaE"

np.random.seed(RANDOM_STATE)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===================== 特征工程函数（不是管道） =====================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """为数据框添加工程特征"""
    df = df.copy()
    
    # 1. 从样品名称提取类别
    df["paint_category"] = df["sample"].apply(_get_paint_category)
    
    # 2. 老化条件数值化
    df["aging_condition"] = df["aging_condition"].map({"UV": 1, "humid-_heat": 0})
    
    # 3. 时间对数变换（处理非线性）
    df["aging_time_log"] = np.log1p(df["aging_time_day"])
    
    # 4. 颜色饱和度（Lab空间中的色度）
    df["color_saturation"] = np.sqrt(df["a0"] ** 2 + df["b0"] ** 2)
    
    # 5. 交互特征：老化时间 × 颜色饱和度
    df["time_saturation_interaction"] = df["aging_time_day"] * df["color_saturation"]
    
    # 6. 颜色复杂度：L与饱和度的乘积
    df["color_complexity"] = df["L0"] * df["color_saturation"]
    
    return df

def _get_paint_category(sample_name: str) -> str:
    """从样品名称推断颜料类别"""
    if "染料" in sample_name:
        return "natural_dye"
    elif "皮纸" in sample_name:
        return "base_paper"
    elif "中国画-" in sample_name:
        return "chinese_paint"
    elif "颜彩-" in sample_name:
        return "japanese_paint"
    elif "矿物颜料-" in sample_name:
        return "mineral_paint"
    else:
        return "other"

# ===================== 数据加载与校验 =====================
def load_and_validate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载并校验数据"""
    logger.info(f"加载训练集: {TRAIN_CSV_PATH}")
    df_train = pd.read_csv(TRAIN_CSV_PATH, encoding="utf-8")
    
    logger.info(f"加载测试集: {TEST_CSV_PATH}")
    df_test = pd.read_csv(TEST_CSV_PATH, encoding="utf-8")
    
    # 检查必需列
    required_train = ["sample", "aging_condition", "aging_time_day", "L0", "a0", "b0", TARGET_COL]
    required_test = ["sample", "aging_condition", "aging_time_day", "L0", "a0", "b0"]
    
    missing_train = [c for c in required_train if c not in df_train.columns]
    if missing_train:
        raise ValueError(f"训练集缺失列: {missing_train}")
    
    missing_test = [c for c in required_test if c not in df_test.columns]
    if missing_test:
        raise ValueError(f"测试集缺失列: {missing_test}")
    
    logger.info(f"✓ 训练集: {len(df_train)} 行")
    logger.info(f"✓ 测试集: {len(df_test)} 行")
    
    # 异常值处理
    df_train = detect_and_handle_outliers(df_train, is_train=True)
    df_test = detect_and_handle_outliers(df_test, is_train=False)
    
    # 平行样差异分析
    _analyze_parallel_samples(df_train)
    
    return df_train, df_test

def _analyze_parallel_samples(df_train: pd.DataFrame):
    """分析平行样的颜色差异"""
    color_cols = ["L0", "a0", "b0"]
    sample_color_std = df_train.groupby("sample")[color_cols].std()
    
    # 检测差异较大的样品
    outlier_samples_L = sample_color_std[sample_color_std["L0"] > 2].index.tolist()
    outlier_samples_ab = sample_color_std[(sample_color_std["a0"] > 1) | (sample_color_std["b0"] > 1)].index.tolist()
    all_suspected = list(set(outlier_samples_L + outlier_samples_ab))
    
    parallel_count = len(sample_color_std[sample_color_std.max(axis=1) > 0.1])
    logger.info(f"平行样数量: {parallel_count} 个样品存在颜色差异")
    
    if all_suspected:
        logger.warning(f"检测到 {len(all_suspected)} 个差异较大的样品（建议人工核查）: {all_suspected[:5]}")
    else:
        logger.info("✓ 所有样品的初始颜色差异在合理范围内")

def detect_and_handle_outliers(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """检测并处理异常值"""
    df = df.copy()
    
    # 定义需要检查异常值的数值列
    numeric_cols = ["aging_time_day", "L0", "a0", "b0"]
    if is_train:
        numeric_cols.append(TARGET_COL)
    
    outliers_count = {}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        # 使用IQR方法检测异常值
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 检测异常值
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        outliers_count[col] = outliers.sum()
        
        if outliers.sum() > 0:
            logger.info(f"检测到 {outliers.sum()} 个异常值在列 '{col}' (IQR方法)")
            
            # 处理异常值：用中位数替换
            median_val = df[col].median()
            df.loc[outliers, col] = median_val
            logger.info(f"✓ 已用中位数 {median_val:.4f} 替换异常值")
    
    total_outliers = sum(outliers_count.values())
    if total_outliers > 0:
        logger.info(f"✓ 异常值处理完成，共处理 {total_outliers} 个异常值")
    else:
        logger.info("✓ 未检测到异常值")
    
    return df

# ===================== 模型构建 =====================
def build_preprocessor(num_features):
    """构建预处理管道"""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), CAT_FEATURES)
        ]
    )

def build_model(model_type: str = "xgb"):
    """构建模型"""
    if model_type == "xgb" and HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        logger.info("✓ 使用 XGBoost 模型")
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        logger.info("✓ 使用 RandomForest 模型")
    
    return model

# ===================== 训练和评估 =====================
def train_and_evaluate(df_train: pd.DataFrame) -> tuple[Pipeline, list]:
    """使用分组K折交叉验证训练模型"""
    logger.info("\n" + "="*50)
    logger.info("开始分组交叉验证训练 (GroupKFold, n_splits=5)")
    logger.info("="*50)
    
    # 应用特征工程到全量训练集
    df_train_fe = add_engineered_features(df_train)
    
    num_features = ORIGIN_NUM_FEATURES + [
        "color_saturation", "aging_time_log", "time_saturation_interaction", "color_complexity"
    ]
    all_features = num_features + CAT_FEATURES
    
    groups = df_train_fe["sample"].values
    X = df_train_fe[all_features]
    y = df_train_fe[TARGET_COL]
    
    gkf = GroupKFold(n_splits=5)
    fold_metrics = []
    
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 构建和训练模型
        preprocessor = build_preprocessor(num_features)
        model = build_model()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_valid)
        
        mae = mean_absolute_error(y_valid, y_pred)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_valid, y_pred)
        
        fold_metrics.append([mae, rmse, r2])
        logger.info(f"[Fold {fold+1}] MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    
    fold_metrics_np = np.array(fold_metrics)
    mean_mae, mean_rmse, mean_r2 = fold_metrics_np.mean(axis=0)
    
    logger.info("\n" + "="*50)
    logger.info("5折平均性能指标")
    logger.info("="*50)
    logger.info(f"Mean MAE:  {mean_mae:.4f}")
    logger.info(f"Mean RMSE: {mean_rmse:.4f}")
    logger.info(f"Mean R²:   {mean_r2:.4f}")
    
    # 在全量训练集上重新训练最终模型
    logger.info("\n训练最终推理模型...")
    
    preprocessor = build_preprocessor(num_features)
    model = build_model()
    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    final_pipeline.fit(X, y)
    
    return final_pipeline, fold_metrics, num_features

# ===================== 推理 =====================
def infer_and_save(pipeline: Pipeline, df_test: pd.DataFrame, num_features):
    """在测试集上推理并保存结果"""
    logger.info("\n推理测试集...")
    
    # 应用特征工程到测试集
    df_test_fe = add_engineered_features(df_test)
    
    all_features = num_features + CAT_FEATURES
    X_test = df_test_fe[all_features]
    
    test_pred = pipeline.predict(X_test)
    
    # 确保输出目录存在
    PRED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存预测结果
    pred_df = pd.DataFrame({TARGET_COL: test_pred})
    pred_df.to_csv(PRED_OUTPUT_CSV, index=False)
    
    logger.info(f"✓ 预测结果已保存: {PRED_OUTPUT_CSV}")
    logger.info(f"  样本数: {len(pred_df)}")
    logger.info(f"  预测范围: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# ===================== 主程序 =====================
if __name__ == "__main__":
    try:
        logger.info("开始颜料老化色差预测")
        
        # 加载数据
        df_train, df_test = load_and_validate_data()
        
        # 训练模型
        final_model, metrics, num_features = train_and_evaluate(df_train)
        
        # 推理和保存
        infer_and_save(final_model, df_test, num_features)
        
        logger.info("\n" + "="*50)
        logger.info("✓ 程序执行完成！")
        logger.info("="*50)
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        sys.exit(1)
