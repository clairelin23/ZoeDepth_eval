import argparse
import glob
import multiprocessing
import os.path
import subprocess

import cv2
import numpy as np
from PIL import Image
from imageio import mimwrite
from tqdm import tqdm

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


def get_gpu(idx):
    # n_device = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
    total = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
    total = total.split("\n")
    total = np.array([int(device_i) for device_i in total])
    used = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Used | grep -o '[0-9]\+'")
    used = used.split("\n")
    used = np.array([int(device_i) for device_i in used])
    used[idx] -= 16
    used = used / total

    return np.argmin(used)


def run(args):
    dataset, exp, folder, split, idx = args
    exp = exp if split is None else f"{exp}/{split}"
    gpu = get_gpu(idx)

    input_dir = folder
    output_dir = os.path.join("outputs", exp)
    if dataset == "egobody":
        sequence = folder.split("/")[-3]
    elif dataset in {"emdb", "tum"}:
        sequence = folder.split("/")[-2]
    else:
        sequence = os.path.basename(folder)

    print("processing", input_dir, output_dir)
    main(input_dir, output_dir, sequence, f"cuda:{gpu}")
    # if main(input_dir, output_dir, sequence, f"cuda:{gpu}"):
    #     os.system(
    #         f"aws s3 cp /home/zhengzhel/outputs/zoedepth/{exp}/{sequence}"
    #         f" s3://adobe-yizhouz/outputs/zoe_depth/{exp}/{sequence} --recursive"
    #     )


def main(input_dir, output_dir, sequence, gpu):
    conf = get_config("zoedepth", "infer", config_version=config)
    # conf = get_config("zoedepth_nk", "infer")
    images = sorted(glob.glob(os.path.join(input_dir, "*.jpg"))) or sorted(glob.glob(os.path.join(input_dir, "*.png")))

    output_dir = os.path.join(output_dir, sequence)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > len(images):
        print("skipping", output_dir)
        return False
    zoe = build_model(conf).to(gpu)
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
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="kitti", choices=["nyu", "kitti"])
    parser.add_argument(
        "--dataset", type=str, default="davis", choices=["3dpw", "davis", "dyna", "egobody", "emdb", "mannequin", "tum", "kitti"]
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=8)
    args = parser.parse_args()

    if args.dataset == "3dpw":
        folders = glob.glob(os.path.join("datasets/3dpw/imageFiles/*"))
    elif args.dataset == "davis":
        folders = glob.glob("datasets/davis/Human/JPEGImages/Full-Resolution/*")
    elif args.dataset == "dyna":
        assert args.split is not None
        folders = glob.glob(os.path.join("datasets/DynaCam/frames", args.split, "*"))
    elif args.dataset == "egobody":
        folders = glob.glob("datasets/egobody/egocentric_color/*/*/PV")
    elif args.dataset == "emdb":
        folders = glob.glob("datasets/emdb/*/*/images")
    elif args.dataset == "mannequin":
        assert args.split is not None
        folders = glob.glob(os.path.join("datasets/MannequinChallenge/frames", args.split, "*"))
    elif args.dataset == "tum":
        folders = glob.glob("datasets/tum/*/rgb")
    elif args.dataset == "kitti":
        folders = glob.glob("datasets/kitti")
        print(folders)
    else:
        raise NotImplementedError
    args.exp = args.dataset if args.exp is None else args.exp

    with multiprocessing.Pool(args.gpu) as pool:
        args = [(args.dataset, args.exp, folder, args.split, idx % args.gpu) for idx, folder in enumerate(folders)]
        results = pool.map_async(run, args, chunksize=1)
        for result in results.get():
            pass
    print("done")
