# Created by fw at 1/1/21

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import yaml

__ALL__ = ["get_logger", "get_saver"]
KEY = "EXPERIMENT"


def get_logger(cfg: OmegaConf) -> LightningLoggerBase:
    if cfg[KEY].LOGGER.VERSION == "MLFlowLogger":
        from pytorch_lightning.loggers import MLFlowLogger

        tags = {"version": cfg[KEY].NAME}
        logger = MLFlowLogger(
            experiment_name=cfg[KEY].LOGGER.EXPERIMENT_NAME,
            tracking_uri=cfg[KEY].LOGGER.TRACKING_URI,
            tags=tags,
        )
    elif cfg[KEY].LOGGER.VERSION == "WandbLogger":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            name=cfg[KEY].NAME,
            project=cfg[KEY].LOGGER.EXPERIMENT_NAME,
            config=yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.FullLoader),
        )
    else:
        assert False, 'only support MLFlow or Wandb'
    return logger


def get_saver(cfg: OmegaConf) -> ModelCheckpoint:
    args = dict(cfg[KEY].SAVER)
    args["FILENAME"] = args["FILENAME"].format(experiment=cfg[KEY].NAME)
    args = {str(k).lower(): v for k, v in args.items()}
    saver = ModelCheckpoint(**args)
    return saver
