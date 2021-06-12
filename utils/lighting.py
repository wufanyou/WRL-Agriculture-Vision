# Created by fw at 12/31/20

from abc import ABC
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf
from utils.model import get_model
from utils.optimizer import get_optimizer
from utils.losses import get_loss
from utils.metrics import get_metric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Union, List
import torch.nn.functional as F

__ALL__ = ["get_lighting"]
KEY = "LIGHTING"


def get_lighting(cfg: OmegaConf) -> LightningModule:
    model = eval(cfg[KEY].VERSION)(cfg)
    return model


class BaseLightingModule(LightningModule):
    r"""BaseLightingModule.

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.metrics = get_metric(cfg)
        self.loss_fn = get_loss(cfg)

    def forward(self, img) -> Tensor:
        output = self.model.forward_dummy(img)
        return output

    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(batch["img"])
        loss = self.loss_fn(y_hat, batch["gt_semantic_seg"], batch["mask"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: Tensor) -> None:
        y_hat = self(batch["img"])
        loss = self.metrics(y_hat, batch["gt_semantic_seg"], batch["mask"])
        # self.log("val_iou", loss)
        return loss

    def validation_epoch_end(self, outputs: list) -> None:
        self.log("val_iou", self.metrics.compute())

    def configure_optimizers(
        self,
    ) -> Union[Tuple[List[Optimizer], Union[List[LambdaLR], List[dict]]], Optimizer]:
        optimizer, scheduler = get_optimizer(self.cfg, self.model)
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer


# class SegFixLightingModule(BaseLightingModule):
#     def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
#         y_hat = self(batch["img"])  # N,1+8,W,H
#         loss = 0
#         loss += F.binary_cross_entropy_with_logits(
#             y_hat[:, 0], batch["depth"], weight=batch["mask"]
#         )
#         batch["dir_deg"][batch["mask"] == 0] = 255
#         loss += F.cross_entropy(y_hat[:, 1:], batch["dir_deg"], ignore_index=255)
#         self.log("train_loss", loss)
#         return loss
#
#     def validation_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
#         y_hat = self(batch["img"])
#         loss = 0
#         loss += F.binary_cross_entropy_with_logits(
#             y_hat[:, 0], batch["depth"], weight=batch["mask"]
#         )
#         batch["dir_deg"][batch["mask"] == 0] = 255
#         loss += F.cross_entropy(y_hat[:, 1:], batch["dir_deg"].long(), ignore_index=255)
#         self.metrics(loss)
#         return loss
#
#     def validation_epoch_end(self, outputs: list) -> None:
#         self.log("val_loss", self.metrics.compute())


class SmpLightingModule(BaseLightingModule):
    def forward(self, img) -> Tensor:
        output = self.model(img)
        return output


class SmpAugLightingModule(BaseLightingModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        self.cls_loss_weight = cfg["LOSS"].CLS_LOSS_WEIGHT

    def forward(self, img) -> Tensor:
        output, _ = self.model(img)
        return output

    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat, aug = self.model(batch["img"])
        loss = self.loss_fn(y_hat, batch["gt_semantic_seg"], batch["mask"])
        loss += self.cls_loss_weight * F.binary_cross_entropy_with_logits(
            aug, (batch["gt_semantic_seg"].sum((2, 3)) > 0).float()
        )
        loss /= 1 + self.cls_loss_weight
        self.log("train_loss", loss)
        return loss


# class SmpAugLightingModuleV2(BaseLightingModule):
#     def __init__(self, cfg: OmegaConf):
#         super().__init__(cfg)
#         self.cls_loss_weight = cfg["LOSS"].CLS_LOSS_WEIGHT
#
#     def forward(self, img) -> Tensor:
#         output, _ = self.model(img)
#         return output
#
#     def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
#         y_hat, cls = self.model(batch["img"])
#         loss = self.loss_fn(y_hat, batch["gt_semantic_seg"], batch["mask"])
#         loss += self.cls_loss_weight * F.cross_entropy(cls, batch["time"])
#         loss /= 1 + self.cls_loss_weight
#         self.log("train_loss", loss)
#         return loss
