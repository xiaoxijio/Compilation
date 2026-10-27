import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils import *
from tqdm import tqdm 
import atexit, argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, required=True, help='board model path')
    parser.add_argument('--board_ip', type=str, required=True, help='board ip')
    parser.add_argument('--port', type=str, required=True, help='board server port')
    # config datasate
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image, directory of images, or txt file with image paths')
    parser.add_argument('--pred_dir', type=str, default="./tmp_res/pred", help='dir of prediction files (txt) cls cx cy w h conf')
    parser.add_argument("--postprocess_type", type=str, required=True, choices=["yolov3", "yolov5_cve", "yolov7", "yolov8"],
                        help="Type of YOLO model output (yolov3, yolov5_cve, yolov7, yolov8).")
    parser.add_argument("--iouconfig", type=float, default=0.5,
                       help="IoU threshold (0.0-1.0)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (0.0-1.0)")
    
    parser.add_argument('--convert_type', type=str, default="Binary",
                       help='Directory to save processed images')
    parser.add_argument('--letter_box', action='store_true', help='open letter_box')
    parser.add_argument('--dtc', action='store_true', help='open dtc')
    parser.add_argument('--resize', type=int, nargs='+', default=None, help='resize shape')
    parser.add_argument('--crop', type=int, nargs='+', default=None, help='crop shape')
    parser.add_argument('--target_shape', type=int, nargs='+', help='target_shape shape')
    parser.add_argument('--scale', type=float, nargs='+', default=[1], help='output scale')
    parser.add_argument('--zp', type=float, nargs='+', default=[0], help='output zp')   
    opt = parser.parse_args()

    image_list = load_image_paths(opt.image_path)
    board_runner = Board_server(opt.board_ip, opt.port, opt.model_path)
    atexit.register(board_runner.cleanup)
    if opt.model_path.endswith(".onnx"):
        opt.convert_type = 'Binary'
        if opt.postprocess_type == "yolov5_cve":
            opt.postprocess_type = "yolov7"
    os.makedirs(opt.pred_dir, exist_ok=True)
    dataset_creater = DatasetCreater(
        target_shape=opt.target_shape,
        resize=opt.resize, 
        crop=opt.crop, 
        dtc=opt.dtc,
        letter_box=opt.letter_box,
        convert_type=opt.convert_type)
    mod_process = Mod(board_runner, dataset_creater, opt.postprocess_type, conf_thresh=opt.conf, iou_thresh=opt.iouconfig)
    for image_path in tqdm(image_list):
        nxywhc_nor = mod_process.forward(image_path)
        pred_txt_path = os.path.join(opt.pred_dir, os.path.basename(image_path)[:-3]+"txt")
        save_target2txt(nxywhc_nor, pred_txt_path) # 后处理结果保存为txt文件
