import os
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0, './')
from datetime import timedelta
from pathlib import Path
from utils import Board_server
from stereo_datasets.easydict import EasyDict
from stereo_datasets.dataset_template import build_transform_by_cfg
from stereo_datasets.dataset_utils.readpfm import readpfm\

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
def stereo_preprocess(data_path):
    sample = get_sample(data_path)
    image1, image2 = sample['left'], sample['right']
    return [image1, image2]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='model board runner')
    # config board 
    parser.add_argument('--board_ip', type=str, required=True, help='board ip')
    parser.add_argument('--port', type=str, required=True, help='board server port')
    # config datasate
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    # config model
    parser.add_argument('--model_path', type=str, required=True, help='board model path')
    args = parser.parse_args()

    board_runner = Board_server(args.board_ip, args.port, args.model_path)

    data_path = Path(args.data_path)
    i = 0
    bar_length = 50  # 进度条长度
    start_time = time.time()
    try:
        root, dirs, files = next(os.walk(data_path))
        total = len(dirs)
        print(f"总共有 {total} 组图片数据")
    except StopIteration:
        print(f"错误：路径 {data_path} 不存在或为空")
        exit(1)
    # os.chmod(data_path, 0o777)

    output_path = data_path.parent / "npu_out"
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print("目录已成功创建")
        except PermissionError:
            print("权限不足，无法创建目录")
        except Exception as e:
            print(f"创建目录时出错：{e}")
    else:
        print("目录已存在，无需创建")
    for item in range(1, total + 1):
        datas = stereo_preprocess(os.path.join(f"{data_path}/{item}/"))
        result = board_runner.infer_once_with_tensor(datas)
        result = np.frombuffer(result, dtype=np.float32).copy()
        print(result)
        result.tofile(f"{output_path}/{item}_0.float.bin")
        i += 1
        # 打印进度条
        progress = (i + 1) / total
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        if i > 0:
            remaining = (elapsed / (i + 1)) * (total - i - 1)
            remaining_str = str(timedelta(seconds=int(remaining)))
        else:
            remaining_str = "?"
        
        sys.stdout.write(
            f"\r[{bar}] {int(progress * 100)}% | "
            f"已用: {elapsed_str} | 剩余: {remaining_str}"
        )
        sys.stdout.flush()
        if i == total: 
            print("\nEvaluate complete.")
            return

if __name__ == '__main__':
    main()