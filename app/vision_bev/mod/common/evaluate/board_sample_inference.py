# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
import mmdet
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from tqdm import tqdm
import numpy as np
import time

from os import path as osp
import copy
from os.path import join


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    # config_dpath = _get_config_directory()
    # config_fpath = join(config_dpath, fname)
    config_fpath = join(fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod

def _get_pts_bbox_head_cfg(fname):
    """Grab configs necessary to create a pts_bbox_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.model.train_cfg.pts))
    test_cfg = mmcv.Config(copy.deepcopy(config.model.test_cfg.pts))

    pts_bbox_head = model.pts_bbox_head
    pts_bbox_head.update(train_cfg=train_cfg)
    pts_bbox_head.update(test_cfg=test_cfg)
    return pts_bbox_head

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default="fastbev", help='config type')
    parser.add_argument('--board_out_path', default="npu_out path", help='board output path')
    parser.add_argument('--out', default="work_dirs/test.pkl", help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=7,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='mAP', 
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def load_bin_to_bchw(bin_path, target_shape, dtype=torch.float32):
    """
    加载二进制文件并转换为BCHW格式的Tensor
    :param bin_path: 二进制文件路径
    :param target_shape: 目标形状 (B, C, H, W)
    :param dtype: 数据类型, 默认float32
    :return: BCHW格式的Tensor
    """

    with open(bin_path, 'rb') as f:
        raw_data = f.read()
    
    # np_array = np.frombuffer(raw_data, dtype=np.float32) 
    np_array = np.fromfile(bin_path, dtype=np.float32) 
    expected_size = np.prod(target_shape)
    
    if len(np_array) != expected_size and -1 not in target_shape:
        raise ValueError(
            f"文件数据量不匹配，预期{expected_size}元素，实际{len(np_array)}元素。"
            f"请检查文件: {bin_path} 是否损坏或目标形状设置是否正确。"
        )
    
    tensor = torch.from_numpy(np_array).type(dtype)
    tensor = tensor.reshape(target_shape)
    return tensor


def bevformer_output_evaluate(board_out_path):
    from mmdet3d.core import LiDARInstance3DBoxes
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d_plugin.datasets import custom_build_dataset
    from mmdet3d_plugin.nms_free_decoder import NMSFreeCoder
    from mmdet3d_plugin.datasets.builder import build_dataloader_3d_plugin

    if mmdet.__version__ > '2.23.0':
        # If mmdet version > 2.23.0, setup_multi_processes would be imported and
        # used from mmdet instead of mmdet3d.
        from mmdet.utils import setup_multi_processes
    else:
        from mmdet3d.utils import setup_multi_processes

    try:
        # If mmdet version > 2.23.0, compat_cfg would be imported and
        # used from mmdet instead of mmdet3d.
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg
    from mmdet3d.core import bbox3d2result

    def get_bboxes(preds_dicts, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        decoder = NMSFreeCoder(post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                max_num=300,
                voxel_size=[0.2, 0.2, 8],
                num_classes=10)

        preds_dicts = decoder.decode(preds_dicts)
        # print('get_bboxes  preds_dicts af : ' ,len(preds_dicts) ,  preds_dicts[0])

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # print('bboxes : ' , bboxes.shape)

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]

            bboxes = LiDARInstance3DBoxes(bboxes, code_size)
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
    cfg = Config.fromfile("configs/bevformer/bevformer-r50-cbgs.py")
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
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
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader_3d_plugin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    outputs = []
    tokens = np.load('tokens.npy')
    for sample_token in tqdm(tokens, total=len(tokens)):
    # for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # sample_token = data['img_metas'][0].data[0][0]['sample_idx']
        sample_frame_dir = f"{board_out_path}/{sample_token}/"
        out_path = sample_frame_dir + 'npu_out/' + sample_token # 注意板端output featuremap的命名
        if not os.path.exists(sample_frame_dir + 'npu_out/'):
            continue
        scores = load_bin_to_bchw(f'{out_path}_0.float.bin', (1, 1, -1, 10)) 
        bboxes = load_bin_to_bchw(f'{out_path}_1.float.bin', (1, 1, -1, 10))
        outs = {
            'all_cls_scores': torch.Tensor(scores),
            'all_bbox_preds': torch.Tensor(bboxes)
        }
        bbox_res = get_bboxes(outs)
        bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_res
            ]
        pts_bbox=dict(pts_bbox=bbox_results[0])
        result_dicts = []
        result_dicts.append(pts_bbox)
        outputs.extend(result_dicts)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
             eval_kwargs.pop(key, None)
    dataset.evaluate(outputs, **eval_kwargs)

def bev__output_evaluate(args):
    from mmdet3d_bevdet.core import LiDARInstance3DBoxes
    from mmdet3d_bevdet.datasets import build_dataloader, build_dataset
    from mmdet3d_bevdet.models.builder import build_head
    if mmdet.__version__ > '2.23.0':
        from mmdet.utils import setup_multi_processes
    else:
        from mmdet3d_bevdet.utils import setup_multi_processes

    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d_bevdet.utils import compat_cfg
    from mmdet3d_bevdet.core import bbox3d2result

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
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

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    bbox_head_cfg = _get_pts_bbox_head_cfg(args.config)
    centerpoint_head = build_head(bbox_head_cfg)
    centerpoint_head.cuda()

    board_out_path = args.board_out_path

    outputs = []
    tokens = np.load('tokens.npy')
    for sample_token in tqdm(tokens, total=len(tokens)):
        sample_frame_dir = f"{board_out_path}/{sample_token}/"
        out_path = sample_frame_dir + '/npu_out/' + sample_token # 注意板端output featuremap的命名

        out0 = load_bin_to_bchw(f'{out_path}_6x3x256x704_0.float.bin', (1, 2, 128, 128)) 
        out1 = load_bin_to_bchw(f'{out_path}_6x3x256x704_1.float.bin', (1, 1, 128, 128))
        out2 = load_bin_to_bchw(f'{out_path}_6x3x256x704_2.float.bin', (1, 3, 128, 128))
        out3 = load_bin_to_bchw(f'{out_path}_6x3x256x704_3.float.bin', (1, 2, 128, 128))
        out4 = load_bin_to_bchw(f'{out_path}_6x3x256x704_4.float.bin', (1, 2, 128, 128))
        out5 = load_bin_to_bchw(f'{out_path}_6x3x256x704_5.float.bin', (1, 10, 128, 128))
        preds_board = [out0, out1, out2, out3, out4, out5]

        dicts = {}
        dict_keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for j in range(len(dict_keys)):
            dicts[dict_keys[j]] = torch.tensor(preds_board[j]).to(device)

        outs=[dicts]
        outs = tuple([outs])
        img_metas = [{'box_type_3d': LiDARInstance3DBoxes}]
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = centerpoint_head.get_bboxes(outs, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_pts
        ]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        outputs.extend(bbox_results)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

def main():
    args = parse_args()
    start_time = time.time()
    if args.config == "bevformer":
        bevformer_output_evaluate(args.board_out_path)
    else:
        bev__output_evaluate(args)
    end_time = time.time()
    print(f"evaluate time:{end_time - start_time}")

if __name__ == '__main__':
    main()
