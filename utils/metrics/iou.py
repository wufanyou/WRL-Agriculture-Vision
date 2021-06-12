# modify from https://github.com/PyTorchLightning/pytorch-lightning/blob/cc40fa306e8cec8822579246c1fd6ca7cef0edc4/pytorch_lightning/metrics/functional/iou.py
from typing import Any, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.metric import Metric


def _confusion_matrix_update(
    preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, num_classes: int
) -> torch.Tensor:
    preds = preds.argmax(1)
    preds = F.one_hot(preds, num_classes).permute([0, 3, 1, 2]).bool()
    target = target.bool()
    preds = preds.logical_or(
        target.logical_and((preds * target).sum(1).bool()[:, None])
    )
    mask = mask.bool()[:, None]
    intersection = preds.logical_and(target).logical_and(mask).sum([0, 2, 3])
    union = preds.logical_or(target).logical_and(mask).sum([0, 2, 3])
    confmat = torch.stack([intersection, union])
    return confmat


def _iou_from_confmat(confmat: torch.Tensor) -> torch.Tensor:
    return (confmat[0] / confmat[1]).mean()


class CustomizeIoU(Metric):
    def __init__(
        self,
        num_classes: int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.num_classes = num_classes
        self.add_state(
            "confmat",
            default=torch.zeros(2, num_classes),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _confusion_matrix_update(preds, target, mask, self.num_classes)
        self.confmat += confmat

        return _iou_from_confmat(confmat + 1e-9)

    def compute(self) -> torch.Tensor:
        """
        Computes confusion matrix
        """
        return _iou_from_confmat(
            self.confmat,
        )
