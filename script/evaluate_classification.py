import requests
import os
import json
import numpy as np
import cv2
import sys
import time
from utils import load_image_paths, DatasetCreater, Board_server, Preprocess_val_bin
from tqdm import tqdm 
import atexit

class Classification():
    def __init__(self):
        self.acc1 = 0
        self.acc5 = 0
        self.infer_times = 0    
        self.start_time = time.time()
        self.goad_dict = self.parse_txt_to_dict(os.path.dirname(os.path.abspath(__file__)) + "/../data/evaluate_dataset/ILSVRC2012/ILSVRC2012_lable_val.txt")

    def parse_txt_to_dict(self, file_path):
        class_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    class_dict[parts[1]] = int(parts[0])
        return class_dict
    
    def get_true_class(self, name):
        return self.goad_dict.get(name, None)
    
    def softmax(self, x):
        # 防止溢出，将输入减去最大值
        x_exp = np.exp(x - np.max(x))
        return x_exp / np.sum(x_exp, axis=-1, keepdims=True)
    def print_result(self):
        acc1 = 0
        acc5 = 0
        if self.infer_times:
            acc1 = f"{self.acc1 / self.infer_times:.6f}"
            acc5 = f"{self.acc5 / self.infer_times:.6f}"
        tqdm.write(f"infer times:{ self.infer_times} Acc1: {acc1}, Acc5: {acc5}, using time {time.time() - self.start_time:.1f}s, top1 num: {self.acc1}, top5 num:{self.acc5}")

    def add_result(self, infer_result, image_path):
        img_name = os.path.basename(image_path)
        img_name = os.path.splitext(img_name)[0]
        # infer_result = self.softmax(infer_result)
        top_indices = np.argsort(infer_result)[-5:][::-1]
        self.infer_times += 1
        real_class = self.get_true_class(img_name)
        if real_class == None:
            print("not find image true class, name:", img_name)
            return
        if real_class == top_indices[0]:
            self.acc1 += 1
            self.acc5 +=1
        elif real_class in top_indices:
            self.acc5 +=1
        if self.infer_times % 500 == 0:
            self.print_result()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='model board runner')
    # config board 
    parser.add_argument('--board_ip', type=str, required=True, help='board ip')
    parser.add_argument('--port', type=str, required=True, help='board server port')
    # config datasate
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image, directory of images, or txt file with image paths')
    # config preprocess
    parser.add_argument('--convert_type', type=str, default="Binary",
                       help='Directory to save processed images')
    parser.add_argument('--letter_box', action='store_true', help='open letter_box')
    parser.add_argument('--dtc', action='store_true', help='open dtc')
    parser.add_argument('--resize', type=int, nargs='+', default=None, help='resize shape')
    parser.add_argument('--crop', type=int, nargs='+', default=None, help='crop shape')
    parser.add_argument('--target_shape', type=int, nargs='+', help='target_shape shape')
    # config model
    parser.add_argument('--model_path', type=str, required=True, help='board model path')
    args = parser.parse_args()

    dataset_creater = DatasetCreater(
        target_shape=args.target_shape,
        resize=args.resize, 
        crop=args.crop, 
        dtc=args.dtc,
        letter_box=args.letter_box,
        convert_type=args.convert_type)

    board_runner = Board_server(args.board_ip, args.port, args.model_path)
    atexit.register(board_runner.cleanup)
    classification =  Classification()
    image_list = load_image_paths(args.image_path)
    input_preprocess = Preprocess_val_bin(args.model_path, args.convert_type).preprocess
    if not image_list:
        print("No valid images found to process.")
        return
    for image_path in tqdm(image_list):
        after_preprocess = input_preprocess(image_path)
        # after_preprocess = dataset_creater.convert(image_path)
        result = board_runner.infer_once_with_tensor(after_preprocess)
        result = np.frombuffer(result, dtype=np.float32).copy()
        if len(result):
            classification.add_result(result, image_path)
        else:
            print("infer err:", image_path)
            break
    classification.print_result()

if __name__ == '__main__':
    main()