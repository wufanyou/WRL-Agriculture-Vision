from typing import Optional, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"

__all__ = ["CustomiseJaccardLoss"]


def soft_jaccard_score(
    output: Tensor,
    target: Tensor,
    mask: Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target * mask, dim=dims)
        cardinality = torch.sum((output + target) * mask, dim=dims)
    else:
        intersection = torch.sum(output * target * mask)
        cardinality = torch.sum((output + target) * mask)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


def to_tensor(x, dtype=None) -> Tensor:
    if isinstance(x, Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


class CustomiseJaccardLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        smooth: float = 100.0,
        eps: float = 1e-7,
        **kwargs,
    ):
        super(CustomiseJaccardLoss, self).__init__()

        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:

        N, C, W, H = pred.shape
        # target N,C,W,H
        # mask N,W,H
        pred = F.logsigmoid(pred).exp()
        dims = (0, 2)

        mask = mask.reshape(N, 1, -1)
        pred = pred.reshape(N, C, -1)
        target = target.reshape(N, C, -1)

        scores = soft_jaccard_score(
            pred,
            target,
            mask,
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = target.sum(dims) > 0
        loss *= mask.float()

        return loss.mean()
