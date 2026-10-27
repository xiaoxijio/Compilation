import argparse

from mmcv import Config
# from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

from tqdm import tqdm
import os

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d_bevdet.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d_bevdet.utils import compat_cfg

from mmdet.datasets import replace_ImageToTensor
import time

os.environ['CUDA_VISIBLE_DEVICES']='1'

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEV')
    parser.add_argument('--config', default="fastbev", help='deploy config')
    parser.add_argument('--save_path', default="preprocessed_imgs/", help='work dir to save file')
    args = parser.parse_args()
    return args

def bevformer_preprocessed_img(save_path, config_path):
    from mmdet3d.datasets import build_dataset
    from mmdet3d_plugin.datasets.builder import build_dataloader_3d_plugin
    print('debug ....: ' ,save_path, config_path)
    cfg = Config.fromfile(config_path)
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    distributed = False
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_3d_plugin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    cnt = 0
    sample_tokens = []
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample_token = data['img_metas'][0].data[0][0]['sample_idx']
        sample_tokens.append(sample_token)

        sample_frame_dir = f"{save_path}/{sample_token}/"
        os.makedirs(sample_frame_dir, exist_ok=True)

        img = data['img'][0].data[0]
        img_numpy = img.numpy()
        img_numpy.tofile(f"{sample_frame_dir}/{sample_token}_6x3x480x800.bin")

        can_bus = data['img_metas'][0].data[0][0]['can_bus']
        can_bus.tofile(f"{sample_frame_dir}/{sample_token}_canbus_1x18.bin")
        cnt += 1
    os.system("cp data/nuscenes/nuscenes_infos_temporal_val.pkl ../../bevformer/tmp_res/quant_dataset/query.npy " + save_path)
    print(f"saved preprocessed data in {save_path}.")

def bev_preprocessed_img(save_path, config_path):
    from mmdet3d_bevdet.datasets import build_dataloader, build_dataset
    cfg = Config.fromfile(config_path)
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
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        
        sample_token = dataset.data_infos[i]['token']
        sample_frame_dir = f"{save_path}/{sample_token}/"
        os.makedirs(sample_frame_dir, exist_ok=True)

        img = data['img_inputs'][0][0][0]
        img_numpy = img.numpy()
        img_numpy.tofile(f"{sample_frame_dir}/{sample_token}_6x3x256x704.bin")
        cnt += 1
    print(f"saved preprocessed data in {save_path}.")


def main():
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.config == "bevformer":
        bevformer_preprocessed_img(args.save_path, "configs/bevformer/bevformer-r50-cbgs.py")
    elif args.config == "bevdet":
        bev_preprocessed_img(args.save_path, "configs/bevdet/bevdet-r50-cbgs.py")
    elif args.config == "fastbev":
        bev_preprocessed_img(args.save_path, "configs/fastbev/fastbev-r50-cbgs.py")

if __name__ == '__main__':

    main()
