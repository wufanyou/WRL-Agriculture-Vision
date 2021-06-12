# %load train.py
from utils import *
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_only
import argparse


@rank_zero_only
def print_cfg(cfg):
    print(cfg)


def main(cfg):
    model = get_lighting(cfg)
    trainer = get_trainer(cfg)
    dataloader = get_dataloader(cfg)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "-c",
        "--config",
        default="config/DeepLabV3Plus-efficientnet-b3.yaml",
        type=str,
    )

    args = parser.parse_args()
    cfg = get_cfg(args.config)
    seed_everything(cfg.SEED)
    print_cfg(cfg)
    main(cfg)
