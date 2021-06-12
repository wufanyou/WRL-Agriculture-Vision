from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_cross_entropy import *
from .lovasz_loss import *
from .hybird import *
from .dice_loss import *
from .focal_loss import *
from .jaccard import *

# from .rmi_loss import *

__ALL__ = ["get_loss"]
KEY = "LOSS"


def get_loss(cfg: OmegaConf) -> nn.Module:
    args = dict(cfg[KEY].ARGS)
    args = {str(k).lower(): v for k, v in args.items()}
    loss_fn = eval(cfg[KEY].VERSION)(**args)
    return loss_fn
