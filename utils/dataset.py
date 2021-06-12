import json

import torch
import glob
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize
from utils.transform import get_transform
import scipy.io as io
import pandas as pd

# type
from PIL import Image
from typing import Dict  # , Tuple, List, Optional
from torch import Tensor
from torch.utils.data import Dataset
from omegaconf import OmegaConf


__ALL__ = ["get_dataset"]

CLASSES = [
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "water",
    "waterway",
    "weed_cluster",
]

KEY = "DATASET"


def get_dataset(cfg: OmegaConf, split: str) -> Dataset:
    dataset = eval(cfg[KEY].VERSION)
    dataset = dataset(cfg, split)
    return dataset


def to_tensor(img) -> Tensor:
    img = torch.tensor(np.array(img), dtype=torch.float)
    img /= 255
    return img


class BaseDataset(Dataset):
    r"""BaseDataset

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf, split: str) -> None:
        assert split in ["train", "test", "val", "semi"]
        data_path = f"{cfg[KEY].PATH}/{split}/"
        file_names = [
            x.split("/")[-1].split(".")[0]
            for x in glob.glob(f"{data_path}/boundaries/*")
        ]
        self.data_path = data_path
        self.filenames = file_names
        self.split = split
        self.transform = get_transform(cfg, split=split)
        self.classes = CLASSES

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        file_name = self.filenames[item]
        images = self._get_images(file_name)

        output = {
            "img": torch.cat(
                [self.transform(images["nir"]), self.transform(images["rgb"])]
            ),
            "mask": to_tensor(images["boundary"]) * to_tensor(images["masks"]),
        }

        if self.split != "test":
            labels = [
                torch.tensor(np.array(label))[None] / 255 for label in images["labels"]
            ]
            labels = torch.cat(labels)
            background = (labels.sum(0) == 0).float()[None]
            labels = torch.cat([background, labels])
            output["gt_semantic_seg"] = labels

        return output

    def _get_images(self, file_name) -> Dict:
        images = {
            "boundary": Image.open(f"{self.data_path}/boundaries/{file_name}.png"),
            "rgb": Image.open(f"{self.data_path}/images/rgb/{file_name}.jpg"),
            "nir": Image.open(f"{self.data_path}/images/nir/{file_name}.jpg"),
            "masks": Image.open(f"{self.data_path}/masks/{file_name}.png"),
        }
        if self.split != "test":
            labels = []
            for c in self.classes:
                labels.append(
                    Image.open(f"{self.data_path}/labels/{c}/{file_name}.png")
                )
            images["labels"] = labels
        return images

    def __len__(self) -> int:
        return len(self.filenames)


class DatasetV3(BaseDataset):
    def __init__(self, cfg: OmegaConf, split: str) -> None:
        super().__init__(cfg, split)
        self.normalzie = Normalize(
            mean=[0.4601, 0.4348, 0.4422, 0.4390], std=[0.1750, 0.1633, 0.1529, 0.1549]
        )

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        file_name = self.filenames[item]
        images = self._get_images(file_name)

        output = {
            "img": self.normalzie(
                torch.cat(
                    [self.transform(images["nir"]), self.transform(images["rgb"])]
                )
            ),
            "mask": to_tensor(images["boundary"]) * to_tensor(images["masks"]),
        }

        if self.split != "test":
            labels = [
                torch.tensor(np.array(label))[None] / 255 for label in images["labels"]
            ]
            labels = torch.cat(labels)
            background = (labels.sum(0) == 0).float()[None]
            labels = torch.cat([background, labels])
            output["gt_semantic_seg"] = labels

        return output



class DatasetV8(DatasetV3):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        file_name = self.filenames[item]
        images = self._get_images(file_name)
        nir = np.array(images["nir"])[:, :, None]
        rgb = np.array(images["rgb"])
        mask = (np.array(images["masks"]) == 255).reshape(512, 512, 1)
        boundary = (np.array(images["boundary"]) == 255).reshape(512, 512, 1)
        mask = np.logical_and(mask, boundary)
        if self.split != "test":
            labels = [
                (np.array(label) == 255)[:, :, None] for label in images["labels"]
            ]
            mask = np.concatenate([mask] + labels, -1)[None]
        else:
            mask = mask[None]

        img, mask_and_seq = self.transform(
            image=np.concatenate([nir, rgb], -1), segmentation_maps=mask
        )
        img = F.to_tensor(img.copy())  # C,H,W
        img = self.normalzie(img)
        mask = torch.tensor(mask_and_seq[0, :, :, 0]).float()
        img = torch.cat([mask[None], img])
        output = {"img": img, "mask": mask}

        if self.split != "test":
            labels = torch.tensor(mask_and_seq[0, :, :, 1:]).permute([2, 0, 1]).float()
            background = (labels.sum(0) == 0).float()[None]
            labels = torch.cat([background, labels])
            output["gt_semantic_seg"] = labels
        return output



