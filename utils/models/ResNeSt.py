# Created by fw at 1/8/21
from utils.models.resnest import resnest200, resnest50, resnest101
import torch.nn as nn
from omegaconf import OmegaConf
from abc import ABC

KEY = "MODEL"
__ALL__ = ["ResNeSt200"]


class ResNeSt200(nn.Module, ABC):
    def __init__(self, cfg: OmegaConf):
        super(ResNeSt200, self).__init__()
        self.cfg = cfg
        model = resnest200(pretrained=False)
        model.conv1 = nn.Conv2d(
            cfg[KEY].INPUT_CHANNEL,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        model.fc = nn.Linear(2048, cfg[KEY].NUM_CLASSES)
        self.model = model

    def forward(self, **kwargs):
        x = kwargs["x"]
        return self.model(x)
