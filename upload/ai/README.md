# AI 颜料老化色差预测项目

## 📘 项目总览

本仓库包含三个主要部分：

1. `baseline_and_data/`：原始 baseline 代码与数据
2. `1/`：迭代 1，强化过拟合检测与模型稳健性
3. `2/`：迭代 2，增加自动超参数调优与异常值处理

## 🧭 迭代顺序说明

- **Base**：`baseline_and_data/` 提供原始训练与测试数据、基础预测脚本和安装说明。
- **Iteration 1**：`1/` 在原始 baseline 之上加入过拟合检测、时间外推验证和防过拟合参数设计。
- **Iteration 2**：`2/` 在迭代 1 的基础上进一步加入自动超参数调优、异常值处理和更丰富的特征工程。

## 📁 目录结构

```
ai/
├── 1/                        # 迭代 1：过拟合检测与稳健性优化
│   ├── line.py
│   └── README.md
├── 2/                        # 迭代 2：智能调优版 baseline
│   ├── baseline_improved.py
│   ├── predict_out.csv
│   ├── README.md
│   └── requirements.txt
└── baseline_and_data/        # 原始 baseline 和赛题数据
    ├── baseline.py
    ├── paint_aging_testset.csv
    ├── paint_aging_trainset.csv
    ├── predict_out.csv
    └── README.md
```

## 🚀 运行建议

### 先从基础版本开始
1. 阅读 `baseline_and_data/README.md`
2. 运行 `baseline_and_data/baseline.py`
3. 比较基础版本输出与后续迭代的改进效果

### 再运行迭代版本
- `1/`：用于验证过拟合控制与时间外推能力
- `2/`：用于探索更强的调优策略和异常值处理

## 📌 版本要点

### `baseline_and_data/`
- 原始 baseline 实现
- 包含 `paint_aging_trainset.csv` 和 `paint_aging_testset.csv`
- 适合作为初赛提交的基础版本或对比标准

### `1/`
- 主要关注模型稳健性与过拟合检测
- 使用 Group KFold 5 折交叉验证
- 添加学习曲线与时间外推评估

### `2/`
- 进一步优化特征工程、调优流程和异常值处理
- 支持自动超参数搜索
- 增强模型泛化能力与报错说明

## 📌 使用顺序建议

1. `baseline_and_data/`（原始 baseline）
2. `1/`（迭代 1，稳健性优化）
3. `2/`（迭代 2，智能调优与增强版）

## 🤝 说明

本仓库适用于比赛实战、模型迭代分析和对比实验。每个子目录均独立可运行，按顺序执行可逐步观察改进效果。