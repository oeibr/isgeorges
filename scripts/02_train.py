"""
训练脚本: ResNet50 二分类 (georges vs non_georges)

整体流程:
    1. 加载划分好的训练集和验证集,构建 DataLoader
    2. 构建 ResNet50 模型(加载 ImageNet 预训练权重),替换分类头为 2 类输出
    3. 配置损失函数(CrossEntropyLoss)、优化器(AdamW)、学习率调度器(CosineAnnealingLR)
    4. 循环训练: 每个 epoch 执行 train → validate → 保存最佳模型 → 早停判断
    5. 训练结束后保存完整的历史日志(JSON)

关键设计:
    - 使用预训练权重做迁移学习,大幅减少训练时间并提升泛化能力
    - 训练集做数据增强(随机裁剪、翻转、颜色抖动),验证集不做增强
    - AdamW 优化器 + CosineAnnealingLR 余弦退火学习率调度
    - 早停机制(patience=10)防止过拟合,基于验证集准确率判断
    - 仅保存验证集表现最好的模型权重(best_model.pth)
"""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径,使 from models.resnet50 可以正确导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.resnet50 import build_model

# ==============================================================================
# 路径配置
# ==============================================================================
ROOT = Path(__file__).resolve().parent.parent      # 项目根目录
DATA_DIR = ROOT / "data"                            # 数据目录(由 01_prepare_data.py 生成)
CKPT_DIR = ROOT / "outputs" / "checkpoints"         # 模型检查点保存目录
LOG_DIR = ROOT / "outputs" / "logs"                 # 训练日志保存目录
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 超参数配置
# ==============================================================================
EPOCHS = 100        # 最大训练轮数
BATCH_SIZE = 32     # 每批处理 32 张图片
LR = 1e-4           # 初始学习率
PATIENCE = 10       # 早停耐心值: 连续 10 轮验证集准确率不提升则停止训练
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择 GPU 或 CPU


def get_transforms():
    """
    返回训练集和验证集各自的数据预处理/增强管道

    训练集 (train):
        - RandomResizedCrop(224): 随机裁剪并缩放到 224x224,增加空间不变性
        - RandomHorizontalFlip(): 随机水平翻转,增加数据多样性
        - ColorJitter(0.2, 0.2, 0.2): 随机调整亮度、对比度、饱和度(±20%),增强颜色鲁棒性
        - ToTensor(): 将 PIL 图片转为 [0,1] 范围的张量
        - Normalize: 使用 ImageNet 统计值做标准化,与预训练权重匹配

    验证集 (val):
        - Resize(256) → CenterCrop(224): 统一缩放到 224x224,不做随机增强
        - 其余与训练集相同
    """
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个完整的 epoch

    Args:
        model: 待训练的神经网络模型
        loader: 训练集 DataLoader,每次返回 (images, labels)
        criterion: 损失函数(CrossEntropyLoss)
        optimizer: 优化器(AdamW)
        device: 运行设备(cpu/cuda)

    Returns:
        (平均loss, 准确率)

    训练步骤详解:
        1. model.train(): 切换到训练模式(启用 Dropout, BatchNorm 使用当前 batch 统计量)
        2. optimizer.zero_grad(): 清空上一轮梯度,防止梯度累积
        3. outputs = model(imgs): 前向传播,得到 logits(未归一化的分数)
        4. loss = criterion(outputs, labels): 计算交叉熵损失
        5. loss.backward(): 反向传播,计算所有参数的梯度
        6. optimizer.step(): 根据梯度更新模型参数
    """
    model.train()                                       # 切换到训练模式
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)  # 数据搬到指定设备
        optimizer.zero_grad()                            # 清空梯度
        outputs = model(imgs)                            # 前向传播
        loss = criterion(outputs, labels)                # 计算损失
        loss.backward()                                  # 反向传播
        optimizer.step()                                 # 更新参数
        # 累加 loss (乘以 batch_size 是因为 criterion 返回的是平均 loss)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()  # 统计预测正确的样本数
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    在验证集上评估模型,不计算梯度

    @torch.no_grad() 装饰器:
        - 禁用梯度计算,节省内存并加速推理
        - 模型处于 eval 模式时不需要梯度,加上此装饰器是最佳实践

    Args:
        model: 神经网络模型
        loader: 验证集 DataLoader
        criterion: 损失函数
        device: 运行设备

    Returns:
        (平均loss, 准确率)
    """
    model.eval()                                        # 切换到评估模式(固定 Dropout, BatchNorm 使用全局统计量)
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)                            # 前向传播
        loss = criterion(outputs, labels)                # 计算损失(不反向传播)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    """训练主函数: 编排数据加载 → 模型构建 → 训练循环 → 模型保存"""
    print(f"Device: {DEVICE}")

    # ==========================================================================
    # 步骤 1: 构建数据集和 DataLoader
    # ==========================================================================
    tfms = get_transforms()
    # ImageFolder 自动按子目录名分配类别标签: georges=0, non_georges=1
    train_ds = datasets.ImageFolder(DATA_DIR / "train", tfms["train"])
    val_ds = datasets.ImageFolder(DATA_DIR / "val", tfms["val"])
    # DataLoader 负责批量加载、打乱(shuffle)、多线程预加载(num_workers)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ==========================================================================
    # 步骤 2: 构建模型、损失函数、优化器、学习率调度器
    # ==========================================================================
    model = build_model().to(DEVICE)                     # ResNet50 预训练模型 + 二分类头
    criterion = nn.CrossEntropyLoss()                    # 交叉熵损失(内置 Softmax)
    # AdamW: 带权重衰减的 Adam 优化器,weight_decay 防止过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # CosineAnnealingLR: 学习率按余弦函数从初始值逐渐衰减到接近 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ==========================================================================
    # 步骤 3: 训练循环
    # ==========================================================================
    history = []             # 记录每个 epoch 的指标
    best_val_acc = 0         # 历史最佳验证集准确率
    no_improve = 0           # 连续未提升的轮数(用于早停)

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        # --- 验证 ---
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        # --- 更新学习率 ---
        scheduler.step()

        # 记录本轮指标
        log = {"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
               "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "lr": round(scheduler.get_last_lr()[0], 6)}
        history.append(log)
        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # --- 检查是否刷新最佳验证准确率 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            # 保存最佳模型 checkpoint(包含模型权重、优化器状态、训练信息)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, CKPT_DIR / "best_model.pth")
            print(f"  → Saved best model (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            # --- 早停判断: 连续 PATIENCE 轮未提升则终止训练 ---
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch} (no improve for {PATIENCE} epochs)")
                break

    # ==========================================================================
    # 步骤 4: 保存训练历史日志
    # ==========================================================================
    with open(LOG_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print("Training done.")


if __name__ == "__main__":
    main()
