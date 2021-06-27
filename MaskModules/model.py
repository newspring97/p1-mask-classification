import torch
import torch.nn as nn


from torchvision import models
import timm
from efficientnet_pytorch import EfficientNet

import numpy as np


class ResNet_Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet_Model, self).__init__()
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_features, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(512, num_classes)
                                 )
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

# EfficientNet from https://github.com/lukemelas/EfficientNet-PyTorch
class EfficientNet_Model(nn.Module):
    def __init__(self, num_classes: int=1000):
        super(EfficientNet_Model, self).__init__()
        model = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = model._fc.in_features
        model._fc = nn.Sequential(nn.Linear(in_features, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, num_classes)
                                 )
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class EfficientNet_B2_Model(nn.Module):
    def __init__(self, num_classes: int=1000):
        super(EfficientNet_B2_Model, self).__init__()
        model = EfficientNet.from_pretrained('efficientnet-b2')
        in_features = model._fc.in_features
        model._fc = nn.Sequential(nn.Linear(in_features, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, num_classes)
                                 )
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ResNext_Model(nn.Module):
    def __init__(self, num_classes: int=1000):
        super(ResNext_Model, self).__init__()
        self.model = timm.create_model('ig_resnext101_32x48d', pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ViT_Model(nn.Module):
    def __init__(self, num_classes: int=1000):
        super(ViT_Model, self).__init__()
        self.model = timm.create_model('vit_base_patch32_224', pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__=="__main__":
    model = ResNet_Model(num_classes=10)
    inputs = np.zeros_like(np.array(range(64*3*7*7)).reshape(64, 3, 7, 7))
    output = model(torch.FloatTensor(inputs))
    print(output.argmax(axis=1))