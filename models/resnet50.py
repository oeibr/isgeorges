"""ResNet50 二分类模型"""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model(num_classes=2):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
