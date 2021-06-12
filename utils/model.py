from typing import Optional
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from .models import *

# from mmcv.utils import Config
# from mmseg.models import build_segmentor
import segmentation_models_pytorch as smp

from pytorch_lightning.utilities import rank_zero_only

__ALL__ = ["get_model"]
KEY = "MODEL"


def load_pretrain_model(
    model: nn.Module, pretrain: Optional[str], remove: Optional[int] = 6
) -> None:
    if pretrain is not None:
        pretrain = torch.load(pretrain, map_location="cpu")
        if "state_dict" in pretrain:
            pretrain = pretrain["state_dict"]
        weight = model.state_dict()
        for k, v in pretrain.items():
            # if remove:
            k = k[remove:]
            if k in weight:
                if v.shape == weight[k].shape:
                    weight[k] = v
        model.load_state_dict(weight)


def replace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU(inplace=True))
        else:
            replace_relu(child)


def get_model(cfg: OmegaConf) -> nn.Module:
    if cfg[KEY].LIBRARY == "mmseg":
        assert False, "The library is not implemented"
    elif cfg[KEY].LIBRARY == "smp":
        model = get_smp_model(cfg)
    elif cfg[KEY].LIBRARY == "customize":
        model = get_customize_model(cfg)
    else:
        assert False, "The library is not implemented"
    return model


# def get_mmseg_model(cfg: OmegaConf) -> nn.Module:
#
#     model_cfg = Config.fromfile(cfg[KEY].CONFIG).model
#     model_cfg["pretrained"] = None
#
#     for module in ["decode_head", "auxiliary_head", "backbone", "neck"]:
#         if module in model_cfg:
#             if "norm_cfg" in model_cfg[module]:
#                 model_cfg[module]["norm_cfg"]["type"] = "BN"
#             elif type(model_cfg[module]) == list:
#                 for part in model_cfg[module]:
#                     part["norm_cfg"]["type"] = "BN"
#
#     if type(model_cfg["decode_head"]) == list:
#         for part in model_cfg["decode_head"]:
#             part["num_classes"] = cfg[KEY].NUM_CLASSES
#     else:
#         model_cfg["decode_head"]["num_classes"] = cfg[KEY].NUM_CLASSES
#
#     model_cfg["backbone"]["in_channels"] = cfg[KEY].IN_CHANNELS
#     model = build_segmentor(model_cfg)
#     load_pretrain_model(model, cfg[KEY].PRETRAINED)
#     return model


def get_smp_model(cfg: OmegaConf) -> nn.Module:

    if cfg[KEY].ARGS:
        args = dict(cfg[KEY].ARGS)
        args = {str(k).lower(): v for k, v in args.items()}
    else:
        args = {}
    cls = eval(f"smp.{cfg[KEY].VERSION}")
    model = cls(
        encoder_name=cfg[KEY].ENCODER,
        classes=cfg[KEY].NUM_CLASSES,
        in_channels=cfg[KEY].IN_CHANNELS,
        aux_params={"classes": cfg[KEY].CLS_HEAD} if cfg[KEY].CLS_HEAD else None,
        **args,
    )
    if cfg[KEY].REPLACE_RELU:
        replace_relu(model)
    load_pretrain_model(model, cfg[KEY].PRETRAINED)
    return model


def get_customize_model(cfg: OmegaConf) -> nn.Module:
    cls = eval(f"{cfg[KEY].VERSION}")
    model = cls(
        encoder_name=cfg[KEY].ENCODER,
        classes=cfg[KEY].NUM_CLASSES,
        in_channels=cfg[KEY].IN_CHANNELS,
        aux_params={"classes": cfg[KEY].NUM_CLASSES} if cfg[KEY].CLS_HEAD else None,
    )
    load_pretrain_model(model, cfg[KEY].PRETRAINED)
    return model
