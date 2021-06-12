import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

__ALL__ = ["MaskBinaryCrossEntropy", "BinaryCrossEntropy"]


class MaskBinaryCrossEntropy(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, **kwargs):
        super(MaskBinaryCrossEntropy, self).__init__()
        self.weight = weight
        if self.weight is not None:
            self.weight = torch.tensor(self.weight, dtype=torch.float)
            self.weight = self.weight.reshape(-1, 1, 1)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask N,W,H
        N, C, W, H = pred.shape
        weight = mask[:, None].expand([N, C, W, H])
        if self.weight is not None:
            weight = weight * self.weight.to(weight.device)
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        return loss


class SoftMaskBinaryCrossEntropy(MaskBinaryCrossEntropy):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        smooth_factor: Optional[float] = None,
        **kwargs
    ):
        super(SoftMaskBinaryCrossEntropy, self).__init__(weight, **kwargs)
        self.smooth_factor = smooth_factor

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask N,W,H
        N, C, W, H = pred.shape
        weight = mask[:, None].expand([N, C, W, H])
        if self.weight is not None:
            weight = weight * self.weight.to(weight.device)
        if self.smooth_factor is not None:
            target = (1 - target) * self.smooth_factor + target * (
                1 - self.smooth_factor
            )
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        return loss


class BinaryCrossEntropy(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None):
        super(BinaryCrossEntropy, self).__init__()
        self.weight = weight
        if self.weight is not None:
            self.weight = torch.tensor(self.weight, dtype=torch.float)
            self.weight = self.weight.reshape(-1, 1, 1)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask N,W,H
        # N, C, W, H = pred.shape
        # weight = mask[:, None].expand([N, C, W, H])
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss


class MaskBinaryCrossEntropyIgnoreIndex(MaskBinaryCrossEntropy):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask N,W,H
        N, C, W, H = pred.shape
        weight = (target != 255).float()
        if self.weight is not None:
            weight = weight * self.weight.to(weight.device)
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        return loss
