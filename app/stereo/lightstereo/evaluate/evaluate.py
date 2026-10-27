
import sys
import os
import argparse
import torch

sys.path.insert(0, './')
from functools import partial
from datasets.metric_per_image import epe_metric, d1_metric, threshold_metric
import numpy as np
from datasets.easydict import EasyDict
from datasets.dataset_template import build_transform_by_cfg
from datasets.dataset_utils.readpfm import readpfm
from PIL import Image

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--board_out_path', type=str, default='./npu_out', help='board output path')
    parser.add_argument('--data_path', type=str, default='./sceneflow_datasets/', help='offline dataset path, include stereo images and gt files')
    parser.add_argument('--img_size', nargs='+', type=int, default=[544, 960], help='should be same with model input size.')

    args = parser.parse_args()

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
    
    if len(np_array) != expected_size:
        raise ValueError(
            f"文件数据量不匹配，预期{expected_size}元素，实际{len(np_array)}元素。"
            f"请检查文件是否损坏或目标形状设置是否正确。"
        )
    
    tensor = torch.from_numpy(np_array).type(dtype)
    tensor = tensor.reshape(target_shape)
    return tensor

def get_sample(sample_path):

    transform_names = [{'NAME': 'RightTopPad', 'SIZE': [ 544, 960 ] }, {'NAME': 'TransposeImage'}, {'NAME': 'ToTensor'}, 
                        {'NAME': 'NormalizeImage', 'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225]}] 
    transform_config = [EasyDict(item) for item in transform_names]
    transform = build_transform_by_cfg(transform_config)
    left_img_path = f"{sample_path}/left_img.png"
    right_img_path = f"{sample_path}/right_img.png"
    disp_img_path = f"{sample_path}/disp_gt.pfm"

    left_img = Image.open(left_img_path).convert('RGB')
    left_img = np.array(left_img, dtype=np.float32)
    right_img = Image.open(right_img_path).convert('RGB')
    right_img = np.array(right_img, dtype=np.float32)
    disp_img = readpfm(disp_img_path)[0].astype(np.float32)
    assert not np.isnan(disp_img).any(), 'disp_img has nan'
    sample = {
        'left': left_img,  # [H, W, 3]
        'right': right_img,  # [H, W, 3]
        'disp': disp_img,  # [H, W]
    }
    sample = transform(sample)
    return sample

@torch.no_grad()
def main():
    args = parse_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metric_func_dict = {
        'epe': epe_metric,
        'd1_all': d1_metric,
        'thres_1': partial(threshold_metric, threshold=1),
        'thres_2': partial(threshold_metric, threshold=2),
        'thres_3': partial(threshold_metric, threshold=3),
    }

    MAX_DISP = 192
    METRIC = ['d1_all', 'epe', 'thres_1', 'thres_2', 'thres_3']
    epoch_metrics = {}
    for k in METRIC:
        epoch_metrics[k] = {'indexes': [], 'values': []}

    h, w = args.img_size[0], args.img_size[1]
    count = len(os.listdir(args.board_out_path))
    print("number of images: ", count)

    for i in range(1, count+1):
        if int(i) % 20 == 0:
            print(f"第 {i} 组数据")
        target_shape = (1, 1, h, w) # (1, 1, 544, 960)
        bin_path = args.board_out_path + f"/{i}_0.float.bin" ###
        np_array = np.fromfile(bin_path, dtype=np.float32) 
        board_out = torch.from_numpy(np_array).type(torch.float32)
        disp_pred = board_out.reshape(target_shape).to(device)
        # print(disp_pred.shape)

        sample = get_sample(os.path.join(f"{args.data_path}/{i}/"))
        disp_gt = sample['disp'].unsqueeze(0).to(device)
        # print(disp_gt.shape)
        mask = (disp_gt < MAX_DISP) & (disp_gt > 0)

        for m in METRIC:
            if m not in metric_func_dict:
                raise ValueError("Unknown metric: {}".format(m))
            metric_func = metric_func_dict[m]
            res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
            epoch_metrics[m]['indexes'].extend([0])
            epoch_metrics[m]['values'].extend(res.tolist())

    results = {}
    for k in epoch_metrics.keys():
        results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

    print(f"metrics: {results}")


if __name__ == '__main__':
    main()
