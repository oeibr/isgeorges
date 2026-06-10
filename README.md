# ResNet50 二分类: Georges vs Non-Georges

基于 PyTorch + ResNet50 (ImageNet 预训练) 的图片二分类任务。

## 架构概述

```
Image (224x224) → ResNet50 (pretrained) → Linear(2048→2) → CrossEntropyLoss
```

- **骨干网络**: ResNet50 (ImageNet 预训练权重)
- **分类头**: 替换最后全连接层为 2 类输出
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-4)
- **学习率调度**: CosineAnnealingLR (T_max=100)
- **早停**: patience=10 (基于 val accuracy)

## 安装

```bash
pip install -r requirements.txt
```

## 运行流程

### 1. 数据划分
```bash
python scripts/01_prepare_data.py
```
按 70/15/15 比例划分 train/val/test，使用软链接避免复制。

### 2. 训练
```bash
python scripts/02_train.py
```
训练最多 100 轮，保存最佳模型到 `outputs/checkpoints/best_model.pth`，
训练日志保存到 `outputs/logs/training_history.json`。

### 3. 评估
```bash
python scripts/03_evaluate.py
```
在 test set 上计算 Accuracy / Precision / Recall / F1 / ROC AUC，
生成混淆矩阵、ROC 曲线、PR 曲线图和错误分类列表。

### 4. 推理
```bash
# 单张图片
python scripts/04_infer.py path/to/image.jpg

# 批量推理
python scripts/04_infer.py path/to/directory/
```

## 目录结构

```
├── georges/              # 原始数据 (正类)
├── non_georges/          # 原始数据 (负类)
├── data/                 # 划分后的数据 (软链接)
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── resnet50.py       # 模型定义
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train.py
│   ├── 03_evaluate.py
│   └── 04_infer.py
└── outputs/
    ├── checkpoints/      # 模型权重
    ├── logs/             # 训练日志
    └── results/          # 评估指标和图表
```

## 实验结果

训练完成后，以下文件将包含实验结果:

| 文件 | 内容 |
|------|------|
| `outputs/logs/training_history.json` | 每轮训练指标 |
| `outputs/checkpoints/best_model.pth` | 最佳模型权重 |
| `outputs/results/metrics.json` | 测试集指标 |
| `outputs/results/confusion_matrix.png` | 混淆矩阵 |
| `outputs/results/roc_curve.png` | ROC 曲线 |
| `outputs/results/pr_curve.png` | Precision-Recall 曲线 |
| `outputs/results/misclassified.json` | 错误分类样本 |

## 可能的改进方向

- 更强的数据增强 (RandAugment, Mixup)
- 学习率 warmup + 更激进的调度策略
- 使用更大模型 (ResNet101, EfficientNet)
- 类别不平衡处理 (focal loss, 加权采样)
- TTA (Test-Time Augmentation) 提升推理准确率
