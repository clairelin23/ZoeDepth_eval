# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm
from zoedepth.data.data_mono_pred_ready import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, \
    ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                                 count_parameters)




if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    print('using mps')
    device = torch.device("mps")

@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.to(device), depth.to(device)
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        pred = sample['pred'].to(device)
        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        def r(m):
            return round(m, round_precision)
    else:
        def r(m):
            return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics



def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.to(device)
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics[
        '#params'] = f"{round(count_parameters(model, include_all=True) / 1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu',
               **kwargs):
    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs,
                 "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default=None,
                        help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                   dataset=dataset, **overwrite_kwargs)
