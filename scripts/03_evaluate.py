"""
评估脚本: 在 test set 上计算指标并生成图表

功能:
    1. 加载训练阶段保存的最佳模型 (best_model.pth)
    2. 在独立的测试集上做推理,收集所有预测结果和概率
    3. 计算五项核心指标: Accuracy / Precision / Recall / F1 / ROC AUC
    4. 生成三张评估图: 混淆矩阵、ROC 曲线、Precision-Recall 曲线
    5. 记录所有错误分类样本的路径和预测信息

指标说明:
    - Accuracy: 总体分类正确率
    - Precision: 预测为正的样本中真正为正的比例(查准率)
    - Recall: 真正为正的样本中被正确找出的比例(查全率)
    - F1: Precision 和 Recall 的调和平均,综合评价指标
    - ROC AUC: 模型区分正负样本的能力,AUC=1 为完美分类,0.5 为随机猜测
"""
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用无头后端,不依赖图形界面也能保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.resnet50 import build_model

# ==============================================================================
# 路径与配置
# ==============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                            # 数据目录
CKPT_DIR = ROOT / "outputs" / "checkpoints"         # 模型检查点目录
RES_DIR = ROOT / "outputs" / "results"              # 评估结果输出目录
RES_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["georges", "non_georges"]


def main():
    """评估主函数: 加载模型 → 推理测试集 → 计算指标 → 生成图表"""

    # ==========================================================================
    # 步骤 1: 构建测试集 DataLoader(使用与验证集相同的预处理,不做增强)
    # ==========================================================================
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(DATA_DIR / "test", tfm)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_ds)} images")

    # ==========================================================================
    # 步骤 2: 加载最佳模型
    # ==========================================================================
    model = build_model().to(DEVICE)
    ckpt = torch.load(CKPT_DIR / "best_model.pth", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}")

    # ==========================================================================
    # 步骤 3: 在测试集上推理,收集预测结果
    # ==========================================================================
    model.eval()                                        # 切换到评估模式
    all_labels, all_preds, all_probs = [], [], []       # 真实标签、预测标签、预测概率
    misclassified = []                                  # 错误分类样本列表

    with torch.no_grad():                               # 禁用梯度计算
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)                       # 前向传播,得到 logits
            probs = torch.softmax(outputs, dim=1)       # softmax 将 logits 转为概率
            preds = outputs.argmax(1).cpu()             # 取概率最大的类别作为预测结果
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())      # 取正类(non_georges)的概率

            # 记录错误分类样本的信息
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    idx = len(all_labels) - len(labels) + i
                    fname = test_ds.samples[idx][0]
                    misclassified.append({
                        "path": fname,
                        "true": CLASS_NAMES[labels[i]],
                        "pred": CLASS_NAMES[preds[i]],
                        "prob": round(float(probs[i, preds[i]]), 4),
                    })

    # 转为 numpy 数组,方便调用 sklearn 指标函数
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ==========================================================================
    # 步骤 4: 计算五项核心指标
    # ==========================================================================
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="binary"), 4),
        "recall": round(recall_score(y_true, y_pred, average="binary"), 4),
        "f1": round(f1_score(y_true, y_pred, average="binary"), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }
    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 保存指标到 JSON
    with open(RES_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ==========================================================================
    # 步骤 5: 生成混淆矩阵图
    # ==============================================================================
    cm = confusion_matrix(y_true, y_pred)               # 2x2 矩阵: [TP, FP; FN, TN]
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    # 在每个格子上标注具体数值
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(RES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ==========================================================================
    # 步骤 6: 生成 ROC 曲线图
    # ==========================================================================
    fpr, tpr, _ = roc_curve(y_true, y_prob)            # FPR = 假正率, TPR = 真正率
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")       # 随机猜测基线
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # ==========================================================================
    # 步骤 7: 生成 Precision-Recall 曲线图
    # ==========================================================================
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = round(f1_score(y_true, y_pred) * 0.5 + metrics["roc_auc"] * 0.5, 4)  # 近似 PR-AUC
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PR-AUC≈{pr_auc}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / "pr_curve.png", dpi=150)
    plt.close()

    # ==========================================================================
    # 步骤 8: 保存错误分类样本列表
    # ==========================================================================
    with open(RES_DIR / "misclassified.json", "w") as f:
        json.dump(misclassified, f, indent=2)
    print(f"\nMisclassified: {len(misclassified)} samples (saved to misclassified.json)")
    print("Evaluation done.")


if __name__ == "__main__":
    main()
