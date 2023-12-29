import argparse
import glob
import os.path

import cv2
import numpy as np
import torch.cuda
from PIL import Image
from imageio import mimwrite
from tqdm import tqdm

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


def main(input_dir, output_dir, config, gpu):
    conf = get_config("zoedepth", "infer", config_version=config)
    zoe = build_model(conf).to(gpu)
    images = sorted(glob.glob(os.path.join(input_dir, "*.jpg"))) or sorted(glob.glob(os.path.join(input_dir, "*.png")))

    sequence = os.path.basename(input_dir)
    output_dir = os.path.join(output_dir, sequence)
    os.makedirs(output_dir, exist_ok=True)

    depths = []
    for image_file in tqdm(images):
        image = Image.open(image_file).convert("RGB")
        depth, _ = zoe.infer_pil(image)
        depths.append(depth)
        depth_file = os.path.basename(image_file).replace(".jpg", ".npy").replace(".png", ".npy")
        np.save(os.path.join(output_dir, depth_file), depth)
    depth_min = min([d.min() for d in depths])
    depth_max = max([d.max() for d in depths])
    shape = np.stack([d.shape for d in depths]).max(0)[::-1]
    depths = [((d - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8) for d in depths]
    depths = np.stack([cv2.resize(d, shape) for d in depths])
    mimwrite(os.path.join(output_dir, sequence + ".mp4"), depths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--config", type=str, default="kitti", choices=["nyu", "kitti"])
    args = parser.parse_args()
    main(args.input, args.output, args.config, "cuda" if torch.cuda.is_available() else "cpu")
