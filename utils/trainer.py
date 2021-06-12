# Created by fw at 1/1/21
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from utils.experiment import *
import os
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_only

__ALL__ = ["get_trainer"]
KEY = "TRAINER"

def get_trainer(cfg: OmegaConf) -> Trainer:
    logger = get_logger(cfg)
    checkpoint_callback = get_saver(cfg)
    args = dict(cfg[KEY])
    args = {str(k).lower(): v for k, v in args.items()}
    args["logger"] = logger
    args["callbacks"] = [checkpoint_callback]
    return Trainer(**args)


# class MyCluster(ClusterEnvironment):
#     def __init__(self):
#         super().__init__()
#         self._master_port = None
#         self._global_rank: int = 0
#         self._world_size: int = 1
#
#     def creates_children(self) -> bool:
#         return False
#
#     def master_address(self) -> str:
#         return os.environ["MASTER_ADDR"]
#
#     def master_port(self) -> int:
#         if self._master_port is None:
#             self._master_port = os.environ["MASTER_PORT"]
#         return int(self._master_port)
#
#     def world_size(self) -> int:
#         return self._world_size
#
#     def set_world_size(self, size: int) -> None:
#         self._world_size = size
#
#     def global_rank(self) -> int:
#         return self._global_rank
#
#     def set_global_rank(self, rank: int) -> None:
#         self._global_rank = rank
#         rank_zero_only.rank = rank
#
#     def local_rank(self) -> int:
#         return int(os.environ.get("LOCAL_RANK", 0))
#
#     def node_rank(self) -> int:
#         group_rank = os.environ.get("GROUP_RANK", 0)
#         return int(os.environ.get("NODE_RANK", group_rank))
#
#     def teardown(self) -> None:
#         if "WORLD_SIZE" in os.environ:
#             del os.environ["WORLD_SIZE"]

# def get_dist_trainer(cfg: OmegaConf) -> Trainer:
#     logger = get_logger(cfg)
#     checkpoint_callback = get_saver(cfg)
#     args = dict(cfg[KEY])
#     args = {str(k).lower(): v for k, v in args.items()}
#     args["logger"] = logger
#     args["callbacks"] = [checkpoint_callback]
#     args["plugins"] = [MyCluster()]
#     return Trainer(**args)
