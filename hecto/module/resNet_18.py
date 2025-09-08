import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=6, pretrained=True):
    # 사전학습된 ResNet18 불러오기
    model = models.resnet18(pretrained=pretrained)

    # 기존 분류기(fc layer)를 새로운 클래스 수에 맞게 교체
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
