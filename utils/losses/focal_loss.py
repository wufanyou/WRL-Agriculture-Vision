from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CustomizeFocalLoss"]


class CustomizeFocalLoss(nn.Module):
    def __init__(
        self, alpha: Optional[float] = 1.0, gamma: Optional[float] = 2.0, **kwargs
    ):
        super(CustomizeFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask N,W,H
        N, C, W, H = pred.shape
        mask = mask[:, None].expand([N, C, W, H])
        pred = torch.sigmoid(pred)
        weight = torch.pow(-pred + 1.0, self.gamma)
        focal = -self.alpha * weight * torch.log(pred)
        loss = (focal * target * mask).mean()
        return loss
