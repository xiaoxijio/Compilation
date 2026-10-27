import os
import sys
import time
import cv2
import os.path as osp
import pickle
import numpy as np
import math
import shutil
from datetime import timedelta
from pathlib import Path
from utils import Board_server
from tqdm import tqdm

import os


def rotate_opencv(img, angle, center):
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(center[0], center[1]), angle=angle, scale=1.0
    )

    rot_img = cv2.warpAffine(
        src=img,
        M=rotation_matrix,
        dsize=(img.shape[1], img.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rot_img


def preProcess(prev_bev, in_angle):  # [2500, 1, 256]
    rotation_angle = in_angle
    tmp_prev_bev = prev_bev.reshape(50, 50, -1)  # [256, 50, 50]

    tmp_prev_bev = rotate_opencv(
        tmp_prev_bev, rotation_angle, center=[100, 100]
    )  # [2500, 1, 256]

    tmp_prev_bev = tmp_prev_bev.reshape(1, 50 * 50, -1)  # [2500, 1, 256]
    prev_bev = tmp_prev_bev  # [2500, 1, 256]

    return prev_bev


def getShift(
    can_bus_af, grid_length=[2 * 51.2 / 50, 2 * 51.2 / 50], bev_h=50, bev_w=50
):
    delta_x = can_bus_af[0]  # np.array([each[0] for each in can_bus_af])
    delta_y = can_bus_af[1]
    ego_angle = can_bus_af[-2] / np.pi * 180
    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]
    translation_length = math.sqrt(delta_x**2 + delta_y**2)
    translation_angle = math.atan2(delta_y, delta_x) / np.pi * 180

    bev_angle = ego_angle - translation_angle
    shift_y = (
        translation_length * math.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    )
    shift_x = (
        translation_length * math.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    )

    shift = np.stack(
        (np.expand_dims(shift_x, 0), np.expand_dims(shift_y, 0)), 0
    ).transpose(1, 0)

    return shift


class ModelRunner:
    def __init__(self, board_ip, port, backbone_path):
        self.backbone_runner = Board_server(board_ip, port, backbone_path)

    def run(self, can_bus, prev_bev_rotate, img, shift):
        result = self.backbone_runner.infer_once_with_tensor(
            [can_bus, prev_bev_rotate, img, shift]
        )
        if len(result) == 3:
            return result
        return np.frombuffer(result, np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="model board runner")
    # config board
    parser.add_argument("--board_ip", type=str, required=True, help="board ip")
    parser.add_argument("--port", type=str, required=True, help="board server port")
    # config datasate
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    # config model
    parser.add_argument(
        "--model_path", type=str, required=True, help="board model path"
    )
    args = parser.parse_args()

    runner = ModelRunner(args.board_ip, args.port, args.model_path)

    data_path = Path(args.data_path)
    start_time = time.time()
    with open(data_path / "nuscenes_infos_temporal_val.pkl", "rb") as f:
        dataset = pickle.load(f)
    new_prev_bev = np.load(os.path.join(data_path, 'query.npy'))
    data_size = len(dataset["infos"])
    img_data_path = data_path
    begin_index = 0
    for idx in tqdm(range(data_size)):
        if idx < begin_index:
            continue
        token = dataset["infos"][idx]["token"]
        cur_image_data_path = img_data_path / token
        img = np.fromfile(
            cur_image_data_path / f"{token}_6x3x480x800.bin", np.float32
        ).reshape(6, 3, 480, 800)
        can_bus = np.fromfile(cur_image_data_path / f"{token}_canbus_1x18.bin").astype(
            np.float32
        )
        if idx == begin_index:
            tmp_pos = can_bus[0:3]
            tmp_angle = can_bus[-1]
        rot_angle = can_bus[-1] - tmp_angle
        prev_bev_rotate = preProcess(new_prev_bev, rot_angle)
        can_bus_onnx = np.copy(can_bus)
        can_bus_onnx[0] = can_bus[0] - tmp_pos[0]
        can_bus_onnx[1] = can_bus[1] - tmp_pos[1]
        can_bus_onnx[2] = can_bus[2] - tmp_pos[2]
        can_bus_onnx[-1] = can_bus[-1] - tmp_angle
        shift = getShift(can_bus_onnx).astype(np.float32)
        result = runner.run(can_bus_onnx, prev_bev_rotate, img, shift)
        if len(result) == 3:
            outfile0, outfile1, new_prev_bev = result
        else: 
            new_prev_bev = result[-2500 * 256 :].reshape(prev_bev_rotate.shape)
            output_data_size = int((len(result) - 2500 * 256) / 2)
            if output_data_size != 3000 and output_data_size != 9000:
                print("get result err. result data size:", len(result))
                break
            outfile0 = result[:output_data_size]
            outfile1 = result[output_data_size:output_data_size*2]
        cur_npu_out_path = img_data_path / token / "npu_out"
        if not cur_npu_out_path.exists():
            cur_npu_out_path.mkdir(parents=True)
        outfile0.tofile(cur_npu_out_path / f"{token}_0.float.bin")
        outfile1.tofile(cur_npu_out_path / f"{token}_1.float.bin")
        tmp_pos = can_bus[0:3]
        tmp_angle = can_bus[-1]
        

if __name__ == "__main__":
    main()
