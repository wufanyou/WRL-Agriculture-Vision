import torch.nn as nn
from torch import Tensor
from typing import Optional
from .lovasz_loss import CustomizeLovaszLoss, LovaszLoss
from .binary_cross_entropy import (
    MaskBinaryCrossEntropyIgnoreIndex,
    MaskBinaryCrossEntropy,
)
from .dice_loss import CustomizeDiceLoss
from .jaccard import CustomiseJaccardLoss
from .focal_loss import CustomizeFocalLoss

__ALL__ = ["Hybird", "HybirdV3", "HybirdV4"]


class Hybird(nn.Module):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super(Hybird, self).__init__()
        self.BCE = MaskBinaryCrossEntropyIgnoreIndex(weight=weight)
        self.lovasz_loss = CustomizeLovaszLoss()
        self.l1 = l1

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        N, C, W, H = pred.shape
        mask = mask[:, None].expand([N, C, W, H])
        target[mask == 0] = 255
        loss = self.BCE(pred, target) + self.l1 * self.lovasz_loss(pred, target)
        loss /= 1 + self.l1
        return loss


class HybirdV3(nn.Module):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super(HybirdV3, self).__init__()
        self.lovasz_loss = LovaszLoss(mode="multiclass", ignore_index=255)
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.l1 = l1

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        target = target.argmax(1)
        target[mask == 0] = 255
        loss = self.ce(pred, target) + self.l1 * self.lovasz_loss(pred, target)
        loss /= 1 + self.l1
        return loss


class HybirdV4(nn.Module):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super(HybirdV4, self).__init__()
        self.bce = MaskBinaryCrossEntropy(weight=weight)
        self.jaccard = CustomiseJaccardLoss(**kwargs)
        self.l1 = l1

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.bce(pred, target, mask) + self.l1 * self.jaccard(pred, target, mask)
        loss /= 1 + self.l1
        return loss


class HybirdV5(nn.Module):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super(HybirdV5, self).__init__()
        self.bce = MaskBinaryCrossEntropy(weight=weight)
        self.dice = CustomizeDiceLoss(**kwargs)
        self.l1 = l1

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.bce(pred, target, mask) + self.l1 * self.dice(pred, target, mask)
        loss /= 1 + self.l1
        return loss


class HybirdV6(nn.Module):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super(HybirdV6, self).__init__()
        self.focal = CustomizeFocalLoss(**kwargs)
        self.jaccard = CustomiseJaccardLoss(**kwargs)
        self.l1 = l1

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.focal(pred, target, mask) + self.l1 * self.jaccard(
            pred, target, mask
        )
        loss /= 1 + self.l1
        return loss
