# Created by fw at 1/6/21

import torch
import torchvision
from utils.models.efficientnet.utils import Swish
import torch.nn as nn
from omegaconf import OmegaConf
from abc import ABC
from collections import OrderedDict


KEY = "MODEL"


def replace_relu(model: nn.Module, activation="ReLU") -> None:
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, eval(activation))
        else:
            replace_relu(child, activation)


class ResNet50(nn.Module, ABC):
    def __init__(self, cfg: OmegaConf):
        super(ResNet50, self).__init__()
        self.cfg = cfg
        model = torchvision.models.resnet18(pretrained=False, progress=False)
        model.fc = nn.Linear(2048, cfg[KEY].NUM_CLASSES)
        self.model = model

    def forward(self, **kwargs):
        x = kwargs["x"]
        return self.model(x)


class ResNet18(nn.Module, ABC):
    def __init__(self, cfg: OmegaConf):
        super(ResNet18, self).__init__()
        self.cfg = cfg
        model = torchvision.models.resnet18(pretrained=False, progress=False)
        model.fc = nn.Linear(512, cfg[KEY].NUM_CLASSES)
        self.model = model

    def forward(self, **kwargs):
        x = kwargs["x"]
        return self.model(x)
