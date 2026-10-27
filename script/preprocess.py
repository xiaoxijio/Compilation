import argparse
import os
import yaml
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from utils import load_image_paths, DatasetCreater, Board_server

# yolov7 yuv420
# python preprocess.py --image_path ./dataset_1.txt --save_path convert_bin/ --letter_box --target_shape 640 640 --convert_type YUV420SP
# yolov7 binary
# python preprocess.py --image_path ./dataset_1.txt --save_path convert_bin/ --letter_box --target_shape 640 640 --convert_type BINARY
# resnet152v1-7 binary
# python preprocess.py --image_path ./dataset_1.txt --save_path convert_bin/ --resize 256 256 --crop 224 224 --target_shape 224 224 --convert_type BINARY --dtc
# resnet152v1-7 yuv420sp
# python preprocess.py --image_path ./dataset_1.txt --save_path convert_bin/ --resize 256 256 --crop 224 224 --target_shape 224 224 --convert_type YUV420SP

def main():
    parser = argparse.ArgumentParser(description='Image preprocessing tool')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image, directory of images, or txt file with image paths')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Directory to save processed images')
    parser.add_argument('--convert_type', type=str, default="Binary", choices=["Binary", "YUV420SP"],
                       help='Directory to save processed images')
    parser.add_argument('--letter_box', action='store_true', help='open letter_box')
    parser.add_argument('--dtc', action='store_true', help='open dtc')
    parser.add_argument('--resize', type=int, nargs='+', default=None, help='resize shape')
    parser.add_argument('--crop', type=int, nargs='+', default=None, help='crop shape')
    parser.add_argument('--target_shape', type=int, nargs='+', help='target_shape shape')
    args = parser.parse_args()


    dataset_creater = DatasetCreater(
        target_shape=args.target_shape,
        resize=args.resize, 
        crop=args.crop, 
        dtc=args.dtc,
        letter_box=args.letter_box,
        convert_type=args.convert_type)

    image_list = load_image_paths(args.image_path)
    os.makedirs(args.save_path, exist_ok=True)
    if not image_list:
        print("No valid images found to process.")
        return
    for image_path in image_list:
        save_path = os.path.join(args.save_path, os.path.basename(image_path))
        save_path = os.path.splitext(save_path)[0]
        result_path = dataset_creater.convert(image_path, save_path)
    print(f"convert {len(image_list)} images to process.")

if __name__ == '__main__':
    main()