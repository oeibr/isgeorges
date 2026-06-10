"""
数据划分脚本: 按 70/15/15 分层划分 train/val/test

功能:
    - 将原始数据 georges/ 和 non_georges/ 按照 70% train / 15% val / 15% test 进行分层划分
    - 使用软链接(os.symlink)而非复制文件,节省磁盘空间
    - 每个类别内部独立划分,保证类别比例一致(分层采样)

输出:
    data/train/georges/      data/train/non_georges/
    data/val/georges/        data/val/non_georges/
    data/test/georges/       data/test/non_georges/
"""
import os
import shutil
import random
from pathlib import Path

# 固定随机种子,保证每次运行划分结果一致
random.seed(42)

# 路径与配置常量
ROOT = Path(__file__).resolve().parent.parent        # 项目根目录
DATA_DIR = ROOT                                       # 原始数据所在目录
OUTPUT_DIR = ROOT / "data"                            # 划分后数据存放目录
CLASSES = ["georges", "non_georges"]                  # 类别名称
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}  # 划分比例


def prepare_data():
    """执行数据划分流程"""
    # 第一步: 创建输出目录结构 data/{split}/{class}/
    for split in SPLIT_RATIO:
        for cls in CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # 第二步: 对每个类别独立划分
    for cls in CLASSES:
        src_dir = DATA_DIR / cls
        # 获取该类别下所有 jpg 文件名,按文件名排序保证确定性
        images = sorted([f.name for f in src_dir.glob("*.jpg")])
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT_RATIO["train"])  # 训练集数量
        n_val = int(n * SPLIT_RATIO["val"])      # 验证集数量

        # 按顺序切片得到三个子集
        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        # 创建软链接到目标目录
        for split, files in splits.items():
            for f in files:
                src = src_dir / f
                dst = OUTPUT_DIR / split / cls / f
                if not dst.exists():
                    os.symlink(src, dst)
            print(f"  {cls}/{split}: {len(files)} images")

    # 第四步: 打印最终划分统计信息
    print("\nData split done:")
    for split in SPLIT_RATIO:
        total = sum(len(list((OUTPUT_DIR / split / c).glob("*.jpg"))) for c in CLASSES)
        print(f"  {split}: {total} images")


if __name__ == "__main__":
    prepare_data()
