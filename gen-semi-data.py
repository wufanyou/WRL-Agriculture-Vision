import glob
import pandas as pd
import numpy as np
from libtiff import TIFF
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing as mp
import multiprocessing.pool as mpp
import os


def gen_dir(path):
    os.makedirs(f"{path}/semi/boundaries", exist_ok=True)
    os.makedirs(f"{path}/semi/images/nir", exist_ok=True)
    os.makedirs(f"{path}/semi/images/rgb", exist_ok=True)
    os.makedirs(f"{path}/semi/labels", exist_ok=True)
    os.makedirs(f"{path}/semi/masks", exist_ok=True)


def process_image(file):
    image = TIFF.open(file).read_image()
    p5 = np.percentile(image, 5)
    p95 = np.percentile(image, 95)
    Vlower = max(
        0, p5 - 0.4 * (p95 - p5)
    )  # The formula is not correct since some raw images may have p5 values > 400
    Vupper = min(255, p95 + 0.4 * (p95 - p5))
    image = np.clip(image, Vlower, Vupper).astype(np.uint8)
    return image


def save_images(data):
    field, flight, files, args = data
    files = files.set_index("channel")
    images = []
    for channel in ["nir", "red", "green", "blue"]:
        images.append(
            process_image(f"{args.in_path}/{field}/{files.loc[channel].filename}")[None]
        )
    images = np.concatenate(images)
    np.random.seed(int(field) + int(flight))  # set seed for different field and flight
    _, W, H = images.shape
    for i in range(args.num_images):
        temp_w = np.random.randint(0, W - 512)
        temp_h = np.random.randint(0, H - 512)
        temp_img = images[:, temp_w : temp_w + 512, temp_h : temp_h + 512]
        filename = (
            f"{field}_{flight}_{i}-{temp_w}-{temp_h}-{temp_w + 512}-{temp_h + 512}"
        )
        nir = Image.fromarray(temp_img[0]).save(
            f"{args.out_path}/semi/images/nir/{filename}.jpg"
        )
        rgb = Image.fromarray(temp_img[1:].swapaxes(0, 1).swapaxes(1, 2)).save(
            f"{args.out_path}/semi/images/rgb/{filename}.jpg"
        )
        # ignore boundary and mask file since we do not know how to process it, set them to one.
        boundary = Image.fromarray((np.ones([512, 512]) * 255).astype(np.uint8))
        boundary.save(f"{args.out_path}/semi/boundaries/{filename}.png")
        boundary.save(f"{args.out_path}/semi/masks/{filename}.png")
    print(field, flight, "finish")


def main(args):
    data = glob.glob(f"{args.in_path}/*/*")
    data = [x.split("/")[-2:] for x in data]
    data = pd.DataFrame(data)
    data.columns = ["field", "filename"]
    data = data.join(
        data["filename"]
        .str.split("_", expand=True)[[0, 1]]
        .rename({0: "flight", 1: "channel"}, axis=1)
    )
    data = data[~data["channel"].isna()].reset_index(drop=True)
    files = []
    for (field, flight), x in data.groupby(["field", "flight"]):
        files.append((field, flight, x, args))
    mpp.Pool(processes=min(mp.cpu_count() // 2, 6)).map(save_images, files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genenate Semi Supervised Data")
    parser.add_argument(
        "-i", "--in-path", default="./raw/", type=str, help="path of raw file"
    )

    parser.add_argument(
        "-o",
        "--out-path",
        default="./supervised/Agriculture-Vision-2021/",
        type=str,
        help="path of output file, should be the same of the ",
    )

    parser.add_argument(
        "-n",
        "--num-images",
        default=300,
        type=int,
        help="number of images per field per flight",
    )

    args = parser.parse_args()
    gen_dir(args.out_path)
    main(args)
