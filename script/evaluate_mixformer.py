import os
import sys
import time
import cv2
import numpy as np
from utils import Board_server
import argparse

GOT10K_VAL_STATE = [
    [178.0, 220.0, 456.0, 315.0],
    [181.0, 157.0, 265.0, 302.0],
    [750.0, 535.0, 157.0, 101.0],
    [1215.0, 139.0, 63.0, 102.0],
    [1031.0, 435.0, 443.0, 324.0],
    [328.0, 294.0, 581.0, 328.0],
    [467.0, 316.0, 456.0, 366.0],
    [885.0, 271.0, 1028.0, 719.0],
    [305.0, 360.0, 580.0, 275.0],
    [119.0, 0.0, 905.0, 574.0],
    [1360.0, 565.0, 319.0, 378.0],
    [372.0, 173.0, 678.0, 478.0],
    [365.0, 231.0, 77.0, 108.0],
    [401.0, 276.0, 101.0, 68.0],
    [189.0, 212.0, 159.0, 129.0],
    [831.0, 216.0, 103.0, 93.0],
    [651.0, 347.0, 539.0, 323.0],
    [952.0, 383.0, 292.0, 242.0],
    [861.0, 390.0, 325.0, 235.0],
    [201.0, 257.0, 34.0, 21.0],
    [57.0, 209.0, 66.0, 38.0],
    [454.0, 238.0, 36.0, 39.0],
    [310.0, 204.0, 65.0, 66.0],
    [991.0, 180.0, 291.0, 210.0],
    [320.0, 253.0, 158.0, 118.0],
    [247.0, 222.0, 121.0, 63.0],
    [812.0, 208.0, 647.0, 275.0],
    [425.0, 144.0, 354.0, 294.0],
    [250.0, 481.0, 474.0, 329.0],
    [788.0, 369.0, 374.0, 361.0],
    [274.0, 166.0, 113.0, 123.0],
    [1012.0, 320.0, 182.0, 315.0],
    [825.0, 523.0, 166.0, 126.0],
    [1023.0, 394.0, 412.0, 187.0],
    [1012.0, 468.0, 69.0, 32.0],
    [705.0, 174.0, 452.0, 415.0],
    [490.0, 167.0, 360.0, 460.0],
    [16.0, 369.0, 497.0, 250.0],
    [490.0, 167.0, 360.0, 460.0],
    [489.0, 173.0, 22.0, 51.0],
    [279.0, 423.0, 1610.0, 355.0],
    [424.0, 434.0, 70.0, 48.0],
    [302.0, 166.0, 517.0, 335.0],
    [504.0, 305.0, 571.0, 139.0],
    [549.0, 246.0, 164.0, 209.0],
    [1008.0, 373.0, 911.0, 351.0],
    [582.0, 271.0, 621.0, 339.0],
    [243.0, 513.0, 985.0, 176.0],
    [808.0, 438.0, 347.0, 390.0],
    [1269.0, 493.0, 61.0, 83.0],
    [799.0, 716.0, 292.0, 233.0],
    [602.0, 236.0, 449.0, 412.0],
    [62.0, 255.0, 397.0, 363.0],
    [1186.0, 254.0, 480.0, 576.0],
    [1054.0, 521.0, 45.0, 36.0],
    [1436.0, 302.0, 38.0, 372.0],
    [982.0, 616.0, 41.0, 156.0],
    [345.0, 0.0, 1574.0, 880.0],
    [528.0, 662.0, 197.0, 193.0],
    [1001.0, 476.0, 271.0, 263.0],
    [1113.0, 455.0, 64.0, 131.0],
    [238.0, 195.0, 69.0, 67.0],
    [469.0, 37.0, 696.0, 653.0],
    [311.0, 195.0, 163.0, 141.0],
    [432.0, 163.0, 1184.0, 738.0],
    [235.0, 273.0, 528.0, 120.0],
    [1045.0, 107.0, 830.0, 639.0],
    [728.0, 380.0, 394.0, 435.0],
    [363.0, 347.0, 232.0, 147.0],
    [597.0, 594.0, 201.0, 259.0],
    [383.0, 346.0, 178.0, 139.0],
    [591.0, 298.0, 310.0, 380.0],
    [247.0, 350.0, 176.0, 107.0],
    [258.0, 97.0, 194.0, 114.0],
    [590.0, 234.0, 158.0, 144.0],
    [632.0, 355.0, 81.0, 103.0],
    [673.0, 546.0, 512.0, 147.0],
    [84.0, 178.0, 103.0, 75.0],
    [1055.0, 432.0, 173.0, 208.0],
    [960.0, 219.0, 188.0, 182.0],
    [445.0, 309.0, 831.0, 437.0],
    [801.0, 240.0, 26.0, 33.0],
    [118.0, 211.0, 88.0, 142.0],
    [506.0, 263.0, 58.0, 75.0],
    [921.0, 242.0, 182.0, 248.0],
    [632.0, 46.0, 402.0, 601.0],
    [465.0, 208.0, 777.0, 654.0],
    [190.0, 229.0, 167.0, 205.0],
    [1112.0, 356.0, 205.0, 489.0],
    [270.0, 38.0, 269.0, 384.0],
    [576.0, 240.0, 128.0, 156.0],
    [504.0, 214.0, 504.0, 486.0],
    [640.0, 590.0, 720.0, 433.0],
    [709.0, 295.0, 831.0, 593.0],
    [541.0, 234.0, 328.0, 262.0],
    [340.0, 266.0, 221.0, 173.0],
    [549.0, 401.0, 157.0, 138.0],
    [183.0, 357.0, 784.0, 356.0],
    [2.0, 0.0, 520.0, 189.0],
    [481.0, 541.0, 860.0, 325.0],
    [605.0, 165.0, 463.0, 489.0],
    [289.0, 240.0, 385.0, 354.0],
    [1003.0, 179.0, 68.0, 82.0],
    [966.0, 808.0, 83.0, 84.0],
    [1019.0, 334.0, 314.0, 209.0],
    [130.0, 49.0, 1663.0, 906.0],
    [416.0, 228.0, 45.0, 45.0],
    [0.0, 0.0, 534.0, 716.0],
    [143.0, 411.0, 1285.0, 411.0],
    [429.0, 64.0, 888.0, 843.0],
    [576.0, 274.0, 122.0, 127.0],
    [781.0, 568.0, 495.0, 364.0],
    [744.0, 497.0, 364.0, 317.0],
    [285.0, 166.0, 1304.0, 473.0],
    [1237.0, 408.0, 376.0, 150.0],
    [465.0, 44.0, 533.0, 571.0],
    [39.0, 398.0, 727.0, 129.0],
    [257.0, 33.0, 782.0, 491.0],
    [671.0, 380.0, 110.0, 156.0],
    [608.0, 469.0, 485.0, 291.0],
    [0.0, 201.0, 1018.0, 854.0],
    [733.0, 315.0, 771.0, 570.0],
    [780.0, 340.0, 466.0, 277.0],
    [754.0, 432.0, 270.0, 170.0],
    [45.0, 80.0, 315.0, 159.0],
    [1270.0, 491.0, 222.0, 195.0],
    [719.0, 547.0, 310.0, 424.0],
    [694.0, 377.0, 54.0, 157.0],
    [458.0, 523.0, 321.0, 142.0],
    [1819.0, 760.0, 98.0, 98.0],
    [303.0, 262.0, 513.0, 210.0],
    [497.0, 232.0, 128.0, 152.0],
    [436.0, 356.0, 56.0, 84.0],
    [280.0, 135.0, 231.0, 123.0],
    [446.0, 279.0, 13.0, 36.0],
    [749.0, 647.0, 290.0, 260.0],
    [271.0, 171.0, 69.0, 77.0],
    [726.0, 399.0, 47.0, 133.0],
    [350.0, 209.0, 880.0, 737.0],
    [852.0, 156.0, 415.0, 521.0],
    [597.0, 309.0, 89.0, 101.0],
    [250.0, 285.0, 800.0, 400.0],
    [240.0, 283.0, 976.0, 661.0],
    [878.0, 420.0, 171.0, 226.0],
    [345.0, 94.0, 101.0, 124.0],
    [848.0, 585.0, 241.0, 292.0],
    [761.0, 547.0, 232.0, 113.0],
    [682.0, 301.0, 239.0, 173.0],
    [663.0, 328.0, 279.0, 191.0],
    [188.0, 107.0, 114.0, 152.0],
    [0.0, 458.0, 1867.0, 180.0],
    [1.0, 170.0, 425.0, 87.0],
    [177.0, 377.0, 1654.0, 414.0],
    [208.0, 396.0, 1710.0, 362.0],
    [655.0, 472.0, 199.0, 145.0],
    [378.0, 74.0, 511.0, 344.0],
    [233.0, 86.0, 103.0, 162.0],
    [593.0, 112.0, 241.0, 411.0],
    [528.0, 312.0, 513.0, 141.0],
    [1925.0, 718.0, 82.0, 179.0],
    [641.0, 470.0, 410.0, 257.0],
    [650.0, 0.0, 344.0, 1078.0],
    [882.0, 324.0, 529.0, 517.0],
    [335.0, 151.0, 301.0, 318.0],
    [942.0, 62.0, 449.0, 716.0],
    [450.0, 362.0, 413.0, 327.0],
    [814.0, 674.0, 162.0, 319.0],
    [790.0, 817.0, 388.0, 198.0],
    [510.0, 273.0, 585.0, 446.0],
    [725.0, 479.0, 275.0, 240.0],
    [382.0, 247.0, 99.0, 49.0],
    [846.0, 352.0, 185.0, 88.0],
    [412.0, 469.0, 201.0, 244.0],
    [626.0, 775.0, 432.0, 251.0],
    [614.0, 353.0, 10.0, 24.0],
    [910.0, 400.0, 260.0, 516.0],
    [920.0, 980.0, 186.0, 236.0],
    [2558.0, 748.0, 518.0, 561.0],
    [434.0, 131.0, 363.0, 264.0],
    [1527.0, 741.0, 812.0, 806.0]
]

class BoardServer:
    def __init__(self, board_ip, port, model_path):
        self.board_runner = Board_server(board_ip, port, model_path)

    def get_inference_result(self, datas):
        result = self.board_runner.infer_once_with_tensor(datas)
        result = np.frombuffer(result, dtype=np.float32).copy()
        return result

class MixformerHead:
    def __init__(self, search_size, parameter_path):
        self.search_size = search_size
        self.dim = 768
        self.parameter_path = parameter_path
    def head_parameter_init(self,):
        with open(os.path.join(self.parameter_path, "boxes_gemm_w1.bin"), 'rb') as f:
            self.boxes_gemm_w1 = f.read()
            self.boxes_gemm_w1 = np.frombuffer(self.boxes_gemm_w1, dtype=np.float32).reshape(768,768)
            self.boxes_gemm_w1 = np.transpose(self.boxes_gemm_w1, (1, 0))
        with open(os.path.join(self.parameter_path, "boxes_gemm_b1.bin"), 'rb') as f:
            self.boxes_gemm_b1 = f.read()
            self.boxes_gemm_b1 = np.frombuffer(self.boxes_gemm_b1, dtype=np.float32).reshape(768,)
        with open(os.path.join(self.parameter_path, "boxes_gemm_w2.bin"), 'rb') as f:
            self.boxes_gemm_w2 = f.read()
            self.boxes_gemm_w2 = np.frombuffer(self.boxes_gemm_w2, dtype=np.float32).reshape(96,768)
            self.boxes_gemm_w2 = np.transpose(self.boxes_gemm_w2, (1, 0))
        with open(os.path.join(self.parameter_path, "boxes_gemm_b2.bin"), 'rb') as f:
            self.boxes_gemm_b2 = f.read()
            self.boxes_gemm_b2 = np.frombuffer(self.boxes_gemm_b2, dtype=np.float32).reshape(96,)
        with open(os.path.join(self.parameter_path, "boxes_layerN_w1.bin"), 'rb') as f:
            self.boxes_layerN_w1 = f.read()
            self.boxes_layerN_w1 = np.frombuffer(self.boxes_layerN_w1, dtype=np.float32).reshape(768,)
        with open(os.path.join(self.parameter_path, "boxes_layerN_b1.bin"), 'rb') as f:
            self.boxes_layerN_b1 = f.read()
            self.boxes_layerN_b1 = np.frombuffer(self.boxes_layerN_b1, dtype=np.float32).reshape(768,)
        with open(os.path.join(self.parameter_path, "boxes_layerN_w2.bin"), 'rb') as f:
            self.boxes_layerN_w2 = f.read()
            self.boxes_layerN_w2 = np.frombuffer(self.boxes_layerN_w2, dtype=np.float32).reshape(96,)
        with open(os.path.join(self.parameter_path, "boxes_layerN_b2.bin"), 'rb') as f:
            self.boxes_layerN_b2 = f.read()
            self.boxes_layerN_b2 = np.frombuffer(self.boxes_layerN_b2, dtype=np.float32).reshape(96,)
        with open(os.path.join(self.parameter_path, "boxes_grid.bin"), 'rb') as f:
            self.boxes_grid = f.read()
            self.boxes_grid = np.frombuffer(self.boxes_grid, dtype=np.float32).reshape(1,96)
        with open(os.path.join(self.parameter_path, "score_gemm_w1.bin"), 'rb') as f:
            self.score_gemm_w1 = f.read()
            self.score_gemm_w1 = np.frombuffer(self.score_gemm_w1, dtype=np.float32).reshape(768,768)
        with open(os.path.join(self.parameter_path, "score_gemm_b1.bin"), 'rb') as f:
            self.score_gemm_b1 = f.read()
            self.score_gemm_b1 = np.frombuffer(self.score_gemm_b1, dtype=np.float32).reshape(768,)
        with open(os.path.join(self.parameter_path, "score_gemm_w2.bin"), 'rb') as f:
            self.score_gemm_w2 = f.read()
            self.score_gemm_w2 = np.frombuffer(self.score_gemm_w2, dtype=np.float32).reshape(768,1)
        with open(os.path.join(self.parameter_path, "score_gemm_b2.bin"), 'rb') as f:
            self.score_gemm_b2 = f.read()
            self.score_gemm_b2 = np.frombuffer(self.score_gemm_b2, dtype=np.float32).reshape(1,)

    def reduce_mean(self, x, axes=None, keepdims=False):
        """
        模仿 ONNX ReduceMean 算子
        :param x: 输入张量
        :param axes: 沿哪些轴求均值（int 或 tuple）
        :param keepdims: 是否保持维度
        :return: 归约后的张量
        """
        return np.mean(x, axis=axes, keepdims=keepdims)

    def reduce_sum(self, x, axes=None, keepdims=True):
        """
        模仿 ONNX ReduceMean 算子
        :param x: 输入张量
        :param axes: 沿哪些轴求均值（int 或 tuple）
        :param keepdims: 是否保持维度
        :return: 归约后的张量
        """
        return np.sum(x, axis=axes, keepdims=keepdims)

    def softmax(self, x, axis=-1):
        """
        对输入 x 沿指定 axis 计算 softmax
        数值稳定：减去最大值
        """
        # 为防止溢出，减去最大值
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def layer_norm(self, x, gamma, beta, eps=1e-5, axis=-1):
        """
        Layer Normalization 实现
        :param x: 输入张量，任意维度
        :param gamma: 缩放参数，形状应与 x 的归一化维度匹配（通常为最后一维）
        :param beta: 偏移参数，形状同 gamma
        :param eps: 防止除零的小常数
        :param axis: 沿哪个轴归一化（默认最后一维）
        :return: 归一化后的张量
        """
        # 计算均值和方差（保持维度以便广播）
        mean = np.mean(x, axis=axis, keepdims=False)
        var = np.var(x, axis=axis, keepdims=False)
        # 归一化
        x_normalized = (x - mean) / np.sqrt(var + eps)
        # 缩放和偏移
        y = gamma * x_normalized + beta

        return y
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def box_compute(self, box_input:np.ndarray):
        box_input = box_input.reshape(1,768)
        box_input = box_input @ self.boxes_gemm_w1 + self.boxes_gemm_b1
        box_input = self.layer_norm(box_input, self.boxes_layerN_w1, self.boxes_layerN_b1)
        box_input = self.relu(box_input)
        box_input = box_input @ self.boxes_gemm_w2 + self.boxes_gemm_b2
        box_input = self.layer_norm(box_input, self.boxes_layerN_w2, self.boxes_layerN_b2)
        box_input = self.softmax(box_input)
        box_input = box_input * self.boxes_grid
        box_input = self.reduce_sum(box_input)
        return box_input

    def tlbr_to_cxywh(self, t, l, b, r, search_size):
        bbox = []
        x1 = t / search_size
        x2 = l / search_size
        y1 = b / search_size
        y2 = r / search_size
        bbox.append((x1 + x2) / 2.0)
        bbox.append((y1 + y2) / 2.0)
        bbox.append(x2 - x1)
        bbox.append(y2 - y1)
        return bbox

    def run_bbox(self, head_input:np.ndarray):
        t = self.box_compute(head_input[0:1, :]).reshape(1,)
        l = self.box_compute(head_input[1:2, :]).reshape(1,)
        b = self.box_compute(head_input[2:3, :]).reshape(1,)
        r = self.box_compute(head_input[3:4, :]).reshape(1,)
        bbox = self.tlbr_to_cxywh(t, l, b, r, self.search_size)
        return bbox

    def run_score(self, head_input:np.ndarray):
        head_input = head_input @ self.score_gemm_w1 + self.score_gemm_b1
        head_input = self.relu(head_input)
        head_input = head_input @ self.score_gemm_w2 + self.score_gemm_b2
        head_input = self.reduce_mean(head_input, axes = 0)
        head_input = head_input.reshape(1)
        return self.sigmoid(head_input)
        
    def head_runner(self, backbone_output:np.ndarray):
        head_input = backbone_output.reshape(298,768)
        head_input = head_input[-4:, :]
        # 计算分数
        score = self.run_score(head_input)
        # 计算bbox
        bbox = self.run_bbox(head_input)
        return score, bbox

class Tracker:
    def __init__(self, board_ip, port, model_path, para_path):
        self.state = None
        self.image = None
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.current_resize_factor = None
        self.max_score_decay = 1.0
        self.max_pred_score = -1.0
        self.current_template = None
        self.online_template = None
        self.online_max_template = None
        self.search_size = 224
        self.search_factor = 4.0
        self.template_factor = 2.0
        self.template_size = 112.0
        self.update_interval = 200
        self.MARGIN = 10
        self.online_size = 1
        self.current_file_name = None
        self.parameter_path = None
        self.board_server = BoardServer(board_ip, port, model_path)
        self.mixformer_head = MixformerHead(self.search_size, para_path)
        self.mixformer_head.head_parameter_init()
        self.out_file_name = None
    def get_sorted_dirs(self, video_path):
        all_dirs = []
        for root, dirs, files in os.walk(video_path):
            for dir_ in dirs:
                full_path = os.path.join(video_path, dir_)
                all_dirs.append(full_path)
        all_dirs.sort()
        return all_dirs

    def get_sorted_image_files(self, directory_path):
        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                full_path = os.path.join(directory_path, file)
                if os.path.isfile(full_path):
                    _, ext = os.path.splitext(full_path)
                    if ext.lower() in image_extensions:
                         image_files.append(full_path)
        # return image_files.sort().reverse()
        image_files.sort()    # sort之后返回的是None, 应该先sort再返回对应的值
        return image_files

    def preprocess(self, sample_target):
        rgb = cv2.cvtColor(sample_target, cv2.COLOR_BGR2RGB)
        float_img = rgb.astype(np.float32) / 255.0
        # print(float_img.shape)
        channels = [
            float_img[:, :, 0],  # R
            float_img[:, :, 1],  # G
            float_img[:, :, 2]   # B
        ]
        h, w = sample_target.shape[:2]
        total_pixels = h * w
        normlized_channels = []
        for c in range(3):
            normalized = (channels[c] - self.mean[c]) / self.std[c]
            normlized_channels.append(normalized)
        normlized_NCHW = np.concatenate([
            normlized_channels[0].flatten(),
            normlized_channels[1].flatten(),
            normlized_channels[2].flatten(),
        ])

        return normlized_NCHW.astype(np.float32)
    def Sample_target(self, search_area_factor: float, output_size: int, search_type: str = "") -> np.ndarray:
        if self.state is None or self.image is None:
            raise Valuerror("state or image not init!")
        x, y, w, h = self.state
        crop_size = np.ceil(np.sqrt(w * h) * search_area_factor)
        if crop_size < 1:
            print("Warning: Too small bound box")
        cx = x + 0.5 * w   # 图像中心 x 坐标
        cy = y + 0.5 * h   # 图像中心 y 坐标

        x1 = int(np.round(x + 0.5 * w - 0.5 * crop_size)) # 增加搜寻区域后的左下角 x 坐标
        y1 = int(np.round(y + 0.5 * h - 0.5 * crop_size)) # 增加搜寻区域后的左下角 y 坐标

        x2 = int(x1 + int(crop_size)) # 增加搜寻区域后的右上角 x 坐标
        y2 = int(y1 + int(crop_size)) # 增加搜寻区域后的右上角 y 坐标
        img_h, img_w = self.image.shape[:2]
        # 计算越界所需的 padding 值
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img_w + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max (y2 - img_h + 1, 0)
        # 计算有效区域坐标
        x1_valid = x1 + x1_pad
        y1_valid = y1 + y1_pad
        x2_valid = x2 - x2_pad
        y2_valid = y2 - y2_pad

        width = x2_valid - x1_valid
        height = y2_valid - y1_valid

        if width <= 0 or height <= 0:
            raise Valuerror("target not exist!")
        y_start = y1 + y1_pad
        x_start =  x1 + x1_pad
        sample_target_crop  = self.image[y_start:y_start + height, x_start:x_start + width].copy()
        sample_target_tmp = cv2.copyMakeBorder(
                sample_target_crop,
                y1_pad, y2_pad, x1_pad, x2_pad,
                borderType=cv2.BORDER_CONSTANT,
            )
        self.current_resize_factor = output_size / crop_size
        # if search_type == "search_area":
        #     print(f"current_resize_factor = {self.current_resize_factor}")
        sample_target = cv2.resize(
            sample_target_tmp,
            (int(output_size), int(output_size)),  # 必须确保这里的参数是整数
            interpolation=cv2.INTER_LINEAR
        )
        return sample_target

    def video_capture_by_frame(self, video_frame_path):
        self.image = cv2.imread(video_frame_path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(f"Could not open or find the image: {video_frame_path}")
        return

    def tracker_info_init(self, first_image_path:str, state_init, image_save_path:str, file_save_path:str, file_name:str):
        self.video_capture_by_frame(first_image_path)
        self.state = state_init
        thickness = 2
        line_type = cv2.LINE_8
        cv2.rectangle(
            self.image,
            (int(self.state[0]), int(self.state[1])),                     # 左上角 (x, y)
            (int(self.state[0] + self.state[2]), int(self.state[1] + self.state[3])), # 右下角 (x+w, y+h)
            (0, 255, 0),  # 绿色 (BGR格式！)
            thickness,
            line_type
        )
        if not os.path.isdir(file_save_path):
            os.makedirs(file_save_path, exist_ok=True)
        # file_path = os.path.join(image_save_path, "0.png")
        # success = cv2.imwrite(file_path, self.image)
        # if not success:
        #     print(f"Failed to save image to {file_path}", file=sys.stderr)
        # print(f"{self.state[0]}, {self.state[1]}, {self.state[2]}, {self.state[3]}")
        self.WIDTH = self.image.shape[1]  # cols = width
        self.HEIGHT = self.image.shape[0]  # rows = height
        sample_target = self.Sample_target(self.template_factor, self.template_size, "template")
        self.current_template = self.preprocess(sample_target)
        self.online_template = self.current_template
        self.online_max_template = self.current_template
        self.current_file_name = file_name
        current_statistic_file = os.path.join(file_save_path, file_name)
        self.out_file_name = current_statistic_file
        with open(self.out_file_name, 'w') as out_file:  # 'w' = truncate mode
            x, y, w, h = self.state
            # 四舍五入取整
            line = f"{int(round(x))} {int(round(y))} {int(round(w))} {int(round(h))}\n"
            out_file.write(line)

    def backbone_inference(self, input_datas) -> np.ndarray:
        result = self.board_server.get_inference_result(input_datas)
        result = np.frombuffer(result, dtype=np.float32).copy()
        return result

    def runner(self, frame_idx, save_path, file_save_path):
        sample_target = self.Sample_target(self.search_factor, self.search_size, "search_area")
        search_area = self.preprocess(sample_target)
        input_datas = [
                        self.current_template,
                        self.online_template,
                        search_area
                    ]
        predict_info = self.backbone_inference(input_datas)
        current_pred_score, bbox = self.mixformer_head.head_runner(predict_info)
        for i in range(len(bbox)):
            bbox[i] = bbox[i] * float(self.search_size) / self.current_resize_factor
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]
        half_side = 0.5 * float(self.search_size) / self.current_resize_factor
        cx_real = cx + (cx_prev - half_side) - 0.5 * w
        cy_real = cy + (cy_prev - half_side) - 0.5 * h
        x1 = cx_real
        y1 = cy_real
        x2 = cx_real + w
        y2 = cy_real + h
        
        x1 = min(max(0.0, x1), float(self.WIDTH - self.MARGIN))
        x2 = min(max(float(self.MARGIN), x2), float(self.WIDTH))
        y1 = min(max(0.0, y1), float(self.HEIGHT - self.MARGIN))
        y2 = min(max(float(self.MARGIN), y2), float(self.HEIGHT))
        w = max(float(self.MARGIN), x2 - x1)
        h = max(float(self.MARGIN), y2 - y1)
        self.state = [x1, y1, w, h]
        with open(self.out_file_name, 'a') as out_file:  # 'w' = truncate mode
            x, y, w, h = self.state
            # 四舍五入取整
            line = f"{int(np.round(x))} {int(np.round(y))} {int(np.round(w))} {int(np.round(h))}\n"
            out_file.write(line)
        self.max_pred_score = self.max_pred_score * self.max_score_decay
        if current_pred_score > 0.5 and current_pred_score > self.max_pred_score:
            template_target = self.Sample_target(self.template_factor, self.template_size, "template")
            self.online_max_template = self.preprocess(template_target)
            self.online_template = self.online_max_template
            self.max_pred_score = current_pred_score
        if frame_idx % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
            self.max_pred_score = -1.0
            self.online_max_template = self.current_template
        thickness = 2
        line_type = cv2.LINE_8
        cv2.rectangle(
            self.image,
            (int(self.state[0]), int(self.state[1])),                     # 左上角 (x, y)
            (int(self.state[0] + self.state[2]), int(self.state[1] + self.state[3])), # 右下角 (x+w, y+h)
            (0, 255, 0),  # 绿色 (BGR格式！)
            thickness,
            line_type
        )
        # os.makedirs(save_path, exist_ok=True)
        # file_path = os.path.join(save_path, f"{frame_idx}.png")
        # success = cv2.imwrite(file_path, self.image)
        # if not success:
        #     print(f"Failed to save image to {file_path}", file=sys.stderr)

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)} min {int(seconds)} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hour {int(minutes)} min"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_ip', type=str, required=True, help='board ip')
    parser.add_argument('--port', type=str, required=True, help='board server port')
    parser.add_argument("--model_path", type=str, required=True, help='board model path')
    parser.add_argument("--video_path", type=str, required=True, help="val video path")
    parser.add_argument("--save_path", type=str, required=True, help="result save path")
    parser.add_argument("--para_path", type=str, required=True, help="mixformer head compute parameter path")
    args = parser.parse_args()

    tracker = Tracker(args.board_ip, args.port, args.model_path, args.para_path)

    val_dirs = tracker.get_sorted_dirs(args.video_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.video_path, exist_ok=True)
    dirs_num = len(val_dirs)
    total_start_time = time.time()
    for k in range(dirs_num):
        val_dir = val_dirs[k]
        images = tracker.get_sorted_image_files(val_dir)
        state_init = GOT10K_VAL_STATE[k]
        current_img_save_path = os.path.join(args.save_path, os.path.basename(val_dir))
        current_file_save_path = args.save_path
        first_image_path = images[0]
        file_name = f"{os.path.basename(val_dir)}.txt"
        tracker.tracker_info_init(first_image_path, state_init, current_img_save_path, current_file_save_path, file_name)
        frame_num = len(images)
        sequence_name = os.path.basename(val_dir)
        start_time = time.time()
        print(f"Processing: {sequence_name}")
        for i in range(1, frame_num):
            video_frame_path = images[i]
            tracker.video_capture_by_frame(video_frame_path)
            tracker.runner(i, current_img_save_path, current_file_save_path)

            if (i + 1) % max(1, (frame_num - 1) // 10) == 0 or i == frame_num - 1:
                progress = (i + 1) / frame_num * 100
                elapsed = time.time() - start_time

                bar = '[' + '#' * int(progress / 10) + '.' * (10 - int(progress / 10)) + ']'
                sys.stdout.write(f'\rProgress: {bar} {progress:5.1f}% '
                           f'({i+1}/{frame_num}) | Time: {elapsed:.1f}s')
                sys.stdout.flush()
        print(f" - Done")

    # 总体统计
    total_time = time.time() - total_start_time
    print(f"All sequences processed.")
    print(f"total time: {format_time(total_time)}")

if __name__ == '__main__':
    main()