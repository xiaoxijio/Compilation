from PIL import Image
import numpy as np
import torchvision
import torch
import json
from pathlib import Path
import os

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))

def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

def sample_augmentation(H, W, final_dim, bot_pct_lim):
    fH, fW = final_dim
    resize = max(fH/H, fW/W)
    resize_dims = (int(W*resize), int(H*resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(bot_pct_lim))*newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    return resize, resize_dims, crop, flip, rotate

def get_preprocessed_sample(imgs_path, H, W, final_dim, bot_pct_lim):
    imgs = []
    for imgname in imgs_path:
        img = Image.open(imgname)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)


        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = sample_augmentation(H, W, final_dim, bot_pct_lim)
        img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                    resize=resize,
                                                    resize_dims=resize_dims,
                                                    crop=crop,
                                                    flip=flip,
                                                    rotate=rotate,
                                                    )
        imgs.append(normalize_img(img))
    return torch.stack(imgs)


def get_calibration_data(json_path, org_dir_path, save_dir_path, H, W, final_dim, bot_pct_lim):
    # 定义要创建的文件夹路径
    save_dir_path = Path(save_dir_path)

    # 创建文件夹，如果已存在则不报错
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # 打开并读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data_1 = data["265f002f02d447ad9074813292eef75e"]

    idx = 0
    B = 1
    N = 6
    C = 3
    for batchi, imgs_path in enumerate(data_1.items()):
        imgs_path = imgs_path[1]
        imgs_path = [os.path.join(org_dir_path, img_path) for img_path in imgs_path]
        sample = get_preprocessed_sample(imgs_path, H, W, final_dim, bot_pct_lim)
        save_path = os.path.join(save_dir_path,f'{idx}_{B}x{N}x{C}x{final_dim[0]}x{final_dim[1]}.bin')
        sample = sample.cpu().numpy()
        sample.tofile(save_path)
        idx = idx + 1
    
    data_2 = data["41fde20fedcd4d22ab26811688612870"]

    for batchi, imgs_path in enumerate(data_2.items()):
        imgs_path = imgs_path[1]
        imgs_path = [os.path.join(org_dir_path, img_path) for img_path in imgs_path]
        sample = get_preprocessed_sample(imgs_path, H, W, final_dim, bot_pct_lim)
        save_path = os.path.join(save_dir_path,f'{idx}_{B}x{N}x{C}x{final_dim[0]}x{final_dim[1]}.bin')
        sample = sample.cpu().numpy()
        sample.tofile(save_path)
        idx = idx + 1

def save_single_input_bin_list(input_dir, output_txt_path):
    """
    单输入场景：将 input_dir 下所有 .bin 文件绝对路径写入 output_txt_path。
    """
    bin_paths = []
    # 需要量化数据数量
    count = 4
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.bin'):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                bin_paths.append(full_path)
                count -= 1
                if count < 1:
                    break
                

    with open(output_txt_path, 'w') as f:
        for path in bin_paths:
            f.write(path + '\n')

    print(f"[单输入] 共保存 {len(bin_paths)} 个 .bin 文件路径至 {output_txt_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='get calibration data')
    parser.add_argument('--pub', type=int, default=0, help='is opt model?')
    parser.add_argument('--quant_data', type=str, required=True, help='data path for quantization')
    parser.add_argument('--out_path', type=str, required=True, help='output path for quantization')
    args = parser.parse_args()

    H = 900
    W = 1600
    bot_pct_lim = (0.0, 0.22)
    final_dim = (256, 704)  # 优化版
    if args.pub == 1:
        print("use opensource onnx model to get calibration data")
        final_dim = (128,352) # 公版

    json_path = args.quant_data + "/nuscenesInfo_64.json"
    org_dir_path = args.quant_data + "/org_data"
    save_dir_path = args.out_path + "/calib_data"
    # get calibration data
    get_calibration_data(json_path, org_dir_path, save_dir_path, H, W, final_dim, bot_pct_lim)
    # save single input bin list
    save_single_input_bin_list(save_dir_path, args.out_path + "/calib_data.txt")


if __name__ == '__main__':
    main()

