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


# # not support onnx torchvision 0.9
# class MobileNetV3Large(nn.Module, ABC):
#     def __init__(self, cfg: OmegaConf):
#         super(MobileNetV3Large, self).__init__()
#         self.cfg = cfg
#         self.model = torchvision.models.mobilenet_v3_large(
#             pretrained=False, progress=False, num_classes=cfg[KEY].NUM_CLASSES
#         )
#
#     def forward(self, **kwargs):
#         x = kwargs["x"]
#         return self.model(x)
#
#
# # not support onnx torchvision 0.9
# class MobileNetV3Small(nn.Module, ABC):
#     def __init__(self, cfg: OmegaConf):
#         super(MobileNetV3Small, self).__init__()
#         self.cfg = cfg
#         self.model = torchvision.models.mobilenet_v3_small(
#             pretrained=False, progress=False, num_classes=cfg[KEY].NUM_CLASSES
#         )
#
#     def forward(self, **kwargs):
#         x = kwargs["x"]
#         return self.model(x)


class MobileNetV2(nn.Module, ABC):
    def __init__(self, cfg: OmegaConf):
        super(MobileNetV2, self).__init__()
        self.cfg = cfg
        self.model = torchvision.models.mobilenet_v2(
            pretrained=False, progress=False, num_classes=cfg[KEY].NUM_CLASSES
        )

    def forward(self, **kwargs):
        x = kwargs["x"]
        return self.model(x)
