import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class MaskCrossEntropy(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None):
        super(MaskCrossEntropy, self).__init__()
        self.weight = weight
        if self.weight is not None:
            self.weight = torch.tensor(self.weight, dtype=torch.float)
            self.weight = self.weight.reshape(-1)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask N,W,H
        N, C, W, H = pred.shape
        target = target.argmax(1)
        target[mask == 0] = 255
        if self.weight is not None:
            self.weight = self.weight.to(pred.device)
        loss = F.cross_entropy(pred, target, ignore_index=255, weight=self.weight)
        return loss
