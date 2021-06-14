# Created by fw at 12/30/20
from omegaconf import OmegaConf
from typing import Optional
__ALL__ = ["get_cfg"]
KEY = "CONFIG"


def get_filename(path: str) -> str:
    filename = path.split("/")[-1].split(".")[0]
    return filename


def get_cfg(path: Optional[str] = None) -> OmegaConf:
    if path is not None:
        cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(_C, cfg)
        cfg.EXPERIMENT.NAME = get_filename(path)
    else:
        cfg = _C.copy()
        cfg.EXPERIMENT.NAME = "NA"
    return cfg


# experiment
_C = OmegaConf.create()
_C.SEED = 2021

_C.EXPERIMENT = OmegaConf.create()
_C.EXPERIMENT.NAME = ""
_C.EXPERIMENT.LOGGER = OmegaConf.create()
_C.EXPERIMENT.LOGGER.VERSION = "MLFlowLogger"
_C.EXPERIMENT.LOGGER.TRACKING_URI = "file:./mlruns"
_C.EXPERIMENT.LOGGER.EXPERIMENT_NAME = "default"

_C.EXPERIMENT.SAVER = OmegaConf.create()
_C.EXPERIMENT.SAVER.MONITOR = "val_loss"
_C.EXPERIMENT.SAVER.VERBOSE = False
_C.EXPERIMENT.SAVER.DIRPATH = "./models/"
_C.EXPERIMENT.SAVER.FILENAME = "{experiment}-{{epoch}}-{{val_iou:.4f}}"
_C.EXPERIMENT.SAVER.SAVE_TOP_K = None
_C.EXPERIMENT.SAVER.SAVE_WEIGHTS_ONLY = True
_C.EXPERIMENT.SAVER.MODE = "max"
_C.EXPERIMENT.SAVER.MONITOR = "val_iou"

# dataset
_C.DATASET = OmegaConf.create()
_C.DATASET.VERSION = "DatasetV8"
_C.DATASET.PATH = "./supervised/Agriculture-Vision-2021/"
# _C.DATASET.SUB_CLASS = None

# data loader
_C.DATALOADER = OmegaConf.create()
_C.DATALOADER.VERSION = "BaseDataLoader"
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.DIMS = (5, 512, 512)
_C.DATALOADER.BATCH_SIZE = OmegaConf.create()
_C.DATALOADER.BATCH_SIZE.TRAIN = 8
_C.DATALOADER.BATCH_SIZE.TEST = 8
_C.DATALOADER.BATCH_SIZE.VAL = 8


# transform
_C.TRANSFORM = OmegaConf.create()
_C.TRANSFORM.VERSION = "TransformV4"
_C.TRANSFORM.RESIZE = 512


# model
_C.MODEL = OmegaConf.create()
_C.MODEL.LIBRARY = "smp"
_C.MODEL.NUM_CLASSES = 9
_C.MODEL.IN_CHANNELS = 5
_C.MODEL.PRETRAINED = None
_C.MODEL.ENCODER = None
_C.MODEL.VERSION = None

# _C.MODEL.CLS_HEAD = False
# _C.MODEL.CONFIG = "./mmseg-configs/hrnet/fcn_hr18_512x512_80k_ade20k.py"

_C.LOSS = OmegaConf.create()
_C.LOSS.VERSION = "HybirdV4"
_C.LOSS.ARGS = OmegaConf.create()
_C.LOSS.ARGS.WEIGHT = None
_C.LOSS.ARGS.L1 = 1.0
_C.LOSS.ARGS.SMOOTH = 100.0

# optimizer
_C.OPTIMIZER = OmegaConf.create()
_C.OPTIMIZER.VERSION = "Adam"

_C.OPTIMIZER.ARGS = OmegaConf.create()
_C.OPTIMIZER.ARGS.LR = 0.0005

_C.OPTIMIZER.SCHEDULER = OmegaConf.create()
_C.OPTIMIZER.SCHEDULER.USE = True
_C.OPTIMIZER.SCHEDULER.VERSION = "linear_schedule_with_warmup"
_C.OPTIMIZER.SCHEDULER.ARGS = OmegaConf.create()
_C.OPTIMIZER.SCHEDULER.ARGS.NUM_WARMUP_STEPS = 1000


# lighting
_C.LIGHTING = OmegaConf.create()
_C.LIGHTING.VERSION = "SmpLightingModule"


_C.METRIC = OmegaConf.create()
_C.METRIC.VERSION = "CustomizeIoU"
_C.METRIC.ARGS = OmegaConf.create()
_C.METRIC.ARGS.NUM_CLASSES = 9

# trainer
_C.TRAINER = OmegaConf.create()
# _C.TRAINER.MAX_STEPS = 500
_C.TRAINER.GPUS = 1
_C.TRAINER.AUTO_SELECT_GPUS = True
