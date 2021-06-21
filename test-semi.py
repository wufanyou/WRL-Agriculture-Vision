from utils import *
from pytorch_lightning import seed_everything
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import glob
import os
import torch
import torchvision.transforms.functional as TF
import scipy.io as io

#

parser = argparse.ArgumentParser(description="Test script")

parser.add_argument(
    "-f",
    "--fold",
    default=0,
    type=int,
)
parser.add_argument(
    "-t",
    "--total-fold",
    default=1,
    type=int,
)

parser.add_argument(
    "-d",
    "--device",
    default="0",
    type=str,
)

parser.add_argument(
    "-c",
    "--config",
    nargs="+",
    default=[
        "./config/DeepLabV3Plus-efficientnet-b3-semi.yaml",  # 0.35
        "./config/DeepLabV3Plus-efficientnet-b5-semi.yaml",  # 0.30
        "./config/FPN-efficientnet-b5-semi.yaml",  # 0.25
        "./config/FPN-efficientnet-b4-semi.yaml",  # 0.10
    ],
    help="Configs path",
)

parser.add_argument("-o", "--out-path", default="./semi-submission", type=str)

args = parser.parse_args()
args.device = f"cuda:{args.fold % 4}"
if args.out_path[-1] != "/":
    args.out_path += "/"


class TTA:
    def __call__(self, x):
        output = torch.cat(
            [
                x,
                TF.vflip(x),
                TF.hflip(x),
                TF.rotate(x, 90),
                TF.rotate(x, -90),
                TF.rotate(x, 180),
            ]
        )
        return output


class ReverseTTA:
    def __call__(self, x):
        output = torch.cat(
            [
                x[0][None],
                TF.vflip(x[1][None]),
                TF.hflip(x[2][None]),
                TF.rotate(x[3][None], -90),
                TF.rotate(x[4][None], 90),
                TF.rotate(x[5][None], 180),
            ]
        )
        return output


def load_weight(model, cfg):
    ckpt = glob.glob(f"{cfg.EXPERIMENT.SAVER.DIRPATH}/{cfg.EXPERIMENT.NAME}-epoch*")[0]
    print("checkpoint", ckpt)
    ckpt = torch.load(ckpt)["state_dict"]
    model.load_state_dict(ckpt)


if __name__ == "__main__":
    seed_everything(2021)

    # DeepLabV3Plus-efficientnet-b3 0.5149
    # DeepLabV3Plus-efficientnet-b5 0.5130
    # FPN-efficientnet-b5 0.5138
    # FPN-efficientnet-b4  0.5100

    cfg = get_cfg(args.config[0])  #
    model = get_lighting(cfg)
    load_weight(model, cfg)
    model = model.eval().to(args.device)
    dataset = get_dataset(cfg, "test")
    dataset.filenames = dataset.filenames[
        args.fold
        * len(dataset.filenames)
        // args.total_fold : (args.fold + 1)
        * len(dataset.filenames)
        // args.total_fold
    ]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    cfg = get_cfg(args.config[1])  # 0.5138
    model_2 = get_lighting(cfg)
    load_weight(model_2, cfg)
    model_2 = model_2.eval().to(args.device)

    cfg = get_cfg(args.config[2])  #
    model_3 = get_lighting(cfg)
    load_weight(model_3, cfg)
    model_3 = model_3.eval().to(args.device)

    cfg = get_cfg(args.config[3])
    model_4 = get_lighting(cfg)
    load_weight(model_4, cfg)
    model_4 = model_4.eval().to(args.device)

    os.makedirs(args.out_path, exist_ok=True)

    tta = TTA()
    reverse_tta = ReverseTTA()

    # Weight [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0] yield  best result for supervised learning track,
    # however we have no chance to submit this version online to the semi-supervised learning track.
    # In order have 100% reproduction, we maintain the our last submission for semi-supervised track (0.520).
    weight = (
        torch.tensor([1, 1, 1, 1.1, 1, 1.4, 1, 1, 1]).reshape(9, 1, 1).to(args.device)
    )

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader)):
            filename = dataloader.dataset.filenames[i] + ".png"
            img = tta(sample["img"].to(args.device))
            mask = sample["mask"].to(args.device)
            _, W, H = mask.shape

            y_hat = model(img).detach()
            y_hat = reverse_tta(y_hat)  # N,C,W,H
            y_hat = torch.sigmoid(y_hat)
            y_hat = y_hat.mean(0)

            y_hat_2 = model_2(img).detach()
            y_hat_2 = reverse_tta(y_hat_2)
            y_hat_2 = torch.sigmoid(y_hat_2)
            y_hat_2 = y_hat_2.mean(0)  # C,W,H

            y_hat_3 = model_3(img).detach()
            y_hat_3 = reverse_tta(y_hat_3)
            y_hat_3 = torch.sigmoid(y_hat_3)
            y_hat_3 = y_hat_3.mean(0)  # C,W,H

            y_hat_4 = model_4(img).detach()
            y_hat_4 = reverse_tta(y_hat_4)
            y_hat_4 = torch.sigmoid(y_hat_4)
            y_hat_4 = y_hat_4.mean(0)  # C,W,H

            y_hat = y_hat * 0.35 + y_hat_2 * 0.3 + y_hat_3 * 0.25 + y_hat_4 * 0.1
            y_hat = y_hat * weight

            y_hat = (y_hat.argmax(0) * mask).detach().cpu().numpy().astype(np.uint8)[0]
            img = Image.fromarray(y_hat)
            img.save(args.out_path + filename)
