import argparse

from mmcv import Config
# from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Dict, Optional, Sequence, Union

import mmcv
import numpy as np

import torch
import tqdm
from mmcv.runner import load_checkpoint
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core import bbox3d2result
import time

os.environ['CUDA_VISIBLE_DEVICES']='1'

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy FastBEV++')
    parser.add_argument('--config', default="FlashOCC_Board/configs/flashocc/flashocc-r50-M0-opt.py", help='deploy config file path')
    parser.add_argument('--save_path', default="preprocessed_imgs/", help='work dir to save file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cnt = 0
    # import ipdb; ipdb.set_trace()
    for i, data in enumerate(data_loader):
        
        sample_token = dataset.data_infos[i]['token']
        sample_frame_dir = f"{args.save_path}/"
        os.makedirs(sample_frame_dir, exist_ok=True)

        img = data['img_inputs'][0][0][0]
        img_numpy = img.numpy()
        img_numpy.tofile(f"{sample_frame_dir}/{sample_token}_6x3x256x704.bin")
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, " / 6019")
    print(f"saved preprocessed data in {args.save_path}.")

if __name__ == '__main__':

    main()
