import requests
import os
import json
import numpy as np
import cv2
from utils import load_image_paths, DatasetCreater, Board_server

# 模型推理单张图片(需要先做好预处理)
# python script/board_server.py \
#     -m app/vision_2d/classification/mobilenetv2/tmp_models/mobilenetv2_yuv420sp.ortm \
#     -i app/vision_2d/classification/mobilenetv2/tmp_res/infer_data/n02009912_262.yuv420sp  \
#     -s ./

# url = "http://10.16.81.27:1234/model"'
def infer_test():
    import argparse
    parser = argparse.ArgumentParser(description='model board runner')
    parser.add_argument('--board_ip', type=str, help='board ip', default="10.16.81.27")
    parser.add_argument('--port', type=str, help='board server port', default="1234")
    parser.add_argument('--model_path', "-m", type=str, required=True, help='board model path')
    parser.add_argument('--input_path', "-i", type=str, required=True, help='input data path for board model')
    parser.add_argument('--save_path', "-s", type=str, default=None, help='output data save path for board model')
    args = parser.parse_args()
    board_runner = Board_server(args.board_ip, args.port, args.model_path)
    result = board_runner.infer_once_with_file(args.input_path)
    if args.save_path!=None:
        result.tofile(args.save_path + "output.bin")

def debug_test():
    import argparse
    parser = argparse.ArgumentParser(description='model board runner')
    parser.add_argument('--board_ip', type=str, help='board ip', default="10.24.65.246")
    parser.add_argument('--port', type=str, help='board server port', default="1234")
    parser.add_argument('--model_path', "-m", type=str, default="data/mnist-12_enable_dump.ortm", help='board model path')
    parser.add_argument('--input_path', "-i", type=str, default="", help='input data path for board model')
    parser.add_argument('--save_path', "-s", type=str, default=None, help='output data save path for board model')
    args = parser.parse_args()
    board_runner = Board_server(args.board_ip, args.port, args.model_path, 30, False)

    # test for get info
    board_runner.get_version()
    board_runner.get_device()

    # test for init_with_perf
    board_runner.init_npu()
    board_runner.loade_board(args.model_path)
    board_runner.get_model_info()
    board_runner.infer_for_perf()
    board_runner.cleanup()
    board_runner.get_info()

    # test for init_with_profiling
    board_runner.init_npu_with_profiling()
    board_runner.loade_board(args.model_path)
    board_runner.get_model_info()
    board_runner.infer_for_profiling()
    board_runner.cleanup()
    board_runner.get_info()

    # test for init_with_dump
    board_runner.init_npu_with_dump()
    board_runner.loade_board(args.model_path)
    board_runner.infer_for_dump(args.input_path)
    board_runner.cleanup()
    board_runner.get_info()


if __name__ == '__main__':
    debug_test()