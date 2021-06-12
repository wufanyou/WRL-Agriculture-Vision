# Created by fw at 12/30/20
import torchvision
import torch
from omegaconf import OmegaConf
from torch import Tensor
import torchvision.transforms.functional as F
import imgaug.augmenters as iaa

__ALL__ = ["get_transform", "get_tta_transform"]
KEY = "TRANSFORM"


def get_transform(cfg: OmegaConf, split: str):
    if split == "semi":
        split = "train"
    assert split in ["train", "test", "val"]
    transform = eval(cfg[KEY].VERSION)(cfg=cfg)(split)
    return transform


def get_tta_transform(version="TTATransform"):
    transform = eval(version)()
    return transform


class BaseTransform:
    r"""BaseTransform.

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg

    @property
    def train_transform(self):
        transform = torchvision.transforms.ToTensor()
        return transform

    @property
    def test_transform(self):
        return self.train_transform

    @property
    def val_transform(self):
        return self.test_transform

    def __call__(self, split: str):
        assert split in ["train", "val", "test"]
        transform = eval(f"self.{split}_transform")
        return transform


class TransformV2(BaseTransform):
    @property
    def train_transform(self):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x * 2 - 1),
            ]
        )
        return transform


class TransformV3(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf(
                        [
                            iaa.VerticalFlip(),
                            iaa.HorizontalFlip(),
                        ]
                    ),
                ),
                iaa.OneOf(
                    [
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ],
            random_order=True,
        )
        return transform


class TransformV4(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.OneOf(
            [
                iaa.VerticalFlip(),
                iaa.HorizontalFlip(),
                iaa.Rot90(0),
                iaa.Rot90(1),
                iaa.Rot90(2),
                iaa.Rot90(3),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Noop()
        return transform


class TransformV5(BaseTransform):
    @property
    def train_transform(self):

        transform = iaa.Sequential(
            [
                iaa.Resize(
                    (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
                    interpolation="nearest",
                ),
                iaa.OneOf(
                    [
                        iaa.VerticalFlip(),
                        iaa.HorizontalFlip(),
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ]
        )

        return transform

    @property
    def test_transform(self):
        transform = iaa.Resize(
            (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
            interpolation="nearest",
        )
        return transform


class TransformV6(BaseTransform):
    @property
    def train_transform(self):

        transform = iaa.Sequential(
            [
                iaa.Resize(
                    (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
                    interpolation="nearest",
                ),
                iaa.Resize((1, 1.25), interpolation="nearest"),
                iaa.Crop(self.cfg[KEY].RESIZE),
                iaa.OneOf(
                    [
                        iaa.VerticalFlip(),
                        iaa.HorizontalFlip(),
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Resize(
            (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
            interpolation="nearest",
        )
        return transform
