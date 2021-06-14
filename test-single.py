from utils import *
from pytorch_lightning import seed_everything
from tqdm import tqdm
import numpy as np
import argparse
import glob
import os
import torch
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description="Test script")
parser.add_argument(
    "-c", "--config", default="config/DeepLabV3Plus-efficientnet-b3.yaml", type=str,
)

parser.add_argument(
    "-f", "--fold", default=0, type=int,
)
parser.add_argument(
    "-t", "--total_fold", default=1, type=int,
)
parser.add_argument(
    "-d", "--device", default='0', type=str,
)

args = parser.parse_args()
args.device = f"cuda:{args.fold}"


class TTA:
    def __call__(self, x):
        output = torch.cat([x, TF.vflip(x), TF.hflip(x), TF.rotate(x, 90), TF.rotate(x, -90), TF.rotate(x, 180)])
        return output


class ReverseTTA:
    def __call__(self, x):
        output = torch.cat([x[0][None],
                            TF.vflip(x[1][None]),
                            TF.hflip(x[2][None]),
                            TF.rotate(x[3][None], -90),
                            TF.rotate(x[4][None], 90),
                            TF.rotate(x[5][None], 180)])
        return output


def load_weight(model, cfg):
    ckpt = glob.glob(
        f"{cfg.EXPERIMENT.SAVER.DIRPATH}/{cfg.EXPERIMENT.NAME}-epoch*"
    )[0]
    print('checkpoint', ckpt)
    ckpt = torch.load(ckpt)["state_dict"]
    model.load_state_dict(ckpt)


if __name__ == '__main__':
    cfg = get_cfg(args.config)
    seed_everything(cfg.SEED)
    model = get_lighting(cfg)
    load_weight(model, cfg)
    model = model.eval().to(args.device)
    dataset = get_dataset(cfg, "test")
    dataset.filenames = dataset.filenames[args.fold * len(dataset.filenames) // args.total_fold:(args.fold + 1) * len(
        dataset.filenames) // args.total_fold]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    os.system(f"mkdir {cfg.EXPERIMENT.NAME}")
    path = f"{cfg.EXPERIMENT.NAME}/"
    tta = TTA()
    reverse_tta = ReverseTTA()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader)):
            filename = dataloader.dataset.filenames[i] + '.png'
            img = tta(sample['img'].to(args.device))
            mask = sample['mask'].to(args.device)
            y_hat = model(img).detach()
            y_hat = reverse_tta(y_hat)  # N,C,W,H
            y_hat = torch.sigmoid(y_hat)
            y_hat = y_hat.mean(0)  # C,W,H
            # print(y_hat.shape,mask.shape)
            y_hat = y_hat.argmax(0) * mask[0]
            y_hat = y_hat.long().detach().cpu()
            y_hat = y_hat.detach().numpy().astype(np.uint8)
            img = Image.fromarray(y_hat)
            img.save(path + filename)