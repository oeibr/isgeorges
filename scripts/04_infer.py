"""
推理脚本: 对单张图片或目录进行预测

功能:
    - 加载训练好的最佳模型 (best_model.pth)
    - 对输入的单张 jpg 图片进行分类预测,输出类别和置信度
    - 也支持输入目录,批量推理目录下所有 jpg 图片

使用方式:
    # 单张图片推理
    python scripts/04_infer.py path/to/image.jpg

    # 批量推理目录下所有图片
    python scripts/04_infer.py path/to/directory/
"""
import sys
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.resnet50 import build_model

# ==============================================================================
# 配置
# ==============================================================================
ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"         # 模型检查点目录
CLASS_NAMES = ["georges", "non_georges"]            # 类别名称(索引 0 → georges, 1 → non_georges)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 图片预处理: 与训练时的验证集预处理保持一致
tfm = transforms.Compose([
    transforms.Resize(256),                         # 短边缩放到 256
    transforms.CenterCrop(224),                     # 中心裁剪到 224x224
    transforms.ToTensor(),                          # 转为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 标准化
])


@torch.no_grad()
def predict(image_path):
    """
    对单张图片进行预测

    Args:
        image_path: 图片文件路径

    Returns:
        (类别名称, 置信度概率)

    推理流程:
        1. 加载模型和最佳权重
        2. 读取图片并做预处理(Resize → CenterCrop → Normalize)
        3. unsqueeze(0) 增加 batch 维度: (3,224,224) → (1,3,224,224)
        4. 前向传播得到 logits,经 softmax 转为概率
        5. 取最大概率对应的类别作为预测结果
    """
    model = build_model().to(DEVICE)
    ckpt = torch.load(CKPT_DIR / "best_model.pth", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img = Image.open(image_path).convert("RGB")     # 确保图片为 RGB 三通道
    inp = tfm(img).unsqueeze(0).to(DEVICE)           # 预处理 + 增加 batch 维度
    out = model(inp)                                 # 前向传播
    probs = torch.softmax(out, dim=1)[0]             # softmax 转概率,取第一个(也是唯一一个)样本
    pred = probs.argmax().item()                     # 取最大概率的类别索引
    return CLASS_NAMES[pred], round(float(probs[pred]), 4)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/04_infer.py <image_path_or_dir>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        # 目录模式: 批量推理该目录下所有 jpg 图片
        for img_path in sorted(path.glob("*.jpg")):
            pred, prob = predict(str(img_path))
            print(f"{img_path.name}: {pred} ({prob})")
    else:
        # 单文件模式: 推理单张图片
        pred, prob = predict(str(path))
        print(f"{path.name}: {pred} ({prob})")
