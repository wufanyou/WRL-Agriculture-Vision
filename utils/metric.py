# Created by fw at 1/1/21
from omegaconf import OmegaConf
import torch.nn as nn
from .metrics import CustomizeIoU

__ALL__ = ["get_metric"]
KEY = "METRIC"


def get_metric(cfg: OmegaConf) -> nn.Module:
    args = dict(cfg[KEY].ARGS)
    args = {str(k).lower(): v for k, v in args.items()}
    loss_fn = eval(cfg[KEY].VERSION)(**args)
    return loss_fn
