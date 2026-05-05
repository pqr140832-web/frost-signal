# 迭代版 1：过拟合检测与稳健性优化

## 分数：19.529027472414263

## 📌 版本说明

这是项目的第一个迭代版本，基于原始 `baseline_and_data` 数据和基础建模流程，重点优化了模型的稳定性与抗过拟合能力。

该版本通过：
- Group KFold 交叉验证
- 训练/验证集过拟合检测
- 学习曲线可视化
- 时间外推验证（短期→长期）
- 限制模型复杂度、防止过拟合的随机森林参数

## 🚀 核心思想

### 1. 过拟合检测
使用 `GroupKFold` 5 折交叉验证，将训练集和验证集的表现进行对比，检测模型是否存在拟合过度问题。

### 2. 时间外推验证
用 `aging_time_day <= 15` 的短期数据训练模型，再对 `aging_time_day > 15` 的长期数据进行测试，评估模型对时间外推的泛化能力。

### 3. 特征工程
- `color_saturation = sqrt(a0^2 + b0^2)`
- `aging_time_log = log1p(aging_time_day)`
- 从 `sample` 名称提取 `paint_category`
- 将 `aging_condition` 编码为数值特征

### 4. 模型设计
使用 `RandomForestRegressor`，并配置防过拟合参数：
- `n_estimators=200`
- `max_depth=5`
- `min_samples_leaf=3`
- `max_features=0.5`

## 📁 文件结构

```
1/
├── line.py             # 主脚本
└── predict_out.csv     # 预测结果输出
```

## ✅ 运行方式

```bash
cd 1
python line.py
```

## 📌 输出

- `1/predict_out.csv`：37 条测试集预测结果
- `learning_curve.png`：学习曲线图，用于直观判断过拟合情况

## ⚠️ 说明

该版本为迭代 1