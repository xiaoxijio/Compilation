import os
import cv2, json
import numpy as np
from pathlib import Path
import requests
import io, re

import time
import torchvision

import cv2 as cv
import numpy as np
import math
import torch

import torch
import onnxruntime as ort
from requests_toolbelt import MultipartEncoder
class Onnx_Runner():
    def __init__(self, model_path:str, device:str = "GPU") -> None:
        if device == "GPU":
            self.__session = ort.InferenceSession(model_path, providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider'])
        else:
            self.__session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = self.__session.get_inputs()
        self.__input_names = [inputs[i].name for i in range(len(inputs))]
        self.__input_shape = [inputs[i].shape for i in range(len(inputs))]
        outputs = self.__session.get_outputs()
        self.__output_names = [outputs[i].name for i in range(len(outputs))]

    def process(self, img):
        if not isinstance(img, list):
            img = [img]
        out = self.__session.run(self.__output_names, {self.__input_names[i]: img[i].reshape(self.__input_shape[i]) for i in range(len(self.__input_names))})
        if len(out) == 1:
            return np.squeeze(out[0])
        return out

def load_image_paths(image_path):
    """Parse image_path argument which could be a single image, directory, or txt file"""
    image_paths = []

    # If it's a text file
    if image_path.endswith('.txt'):
        with open(image_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    image_paths.append(str(line))
    # If it's a directory
    elif os.path.isdir(image_path):
        image_paths = [str(file.resolve()) for file in Path(image_path).rglob("*") if file.suffix.lower() in [".jpg", ".jpeg"]]

    # If it's a single image file
    elif os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        image_paths.append(str(image_path))
    else:
        raise ValueError(f"Invalid image_path: {image_path}. Must be jpg image, directory, or txt file.")

    return image_paths

def yuv420sp(img, h, w):
    m = np.array(
        [
            [1024, 0, 1436],
            [1024, -352, -731],
            [1024, 1815, 0],
        ],
    )
    in_dc =np.array( [[0, 128, 128]])
    m_i = np.linalg.inv(m)
    img = cv2.resize(img, (w, h))
    img = np.transpose(img, axes=[2, 0, 1])
    yuv = np.matmul(m_i, img.reshape([3,-1]))
    yuv = yuv * 1024
    yuv = np.transpose(yuv, axes=[1, 0])

    yuv = yuv + in_dc
    yuv = np.transpose(yuv, axes=[1, 0])
    yuv = yuv.reshape([3, h, w])
    y = yuv[0, :, :]
    u = yuv[1, :, :][::2, ::2]
    v = yuv[2, :, :][::2, ::2]
    uv = np.stack([u,v], axis=2)
    if w % 2:
        uv = uv.reshape([-1, w + 1])
        uv = uv[:, :w]
    uv = uv.reshape([-1, w])
    nv12 = np.concatenate([y,uv], axis=0)
    nv12 = np.clip(nv12, 0, 255).astype(np.uint8)
    return nv12

class DatasetCreater():
    def __init__(self, target_shape, dtc,
            resize = None, crop = None,
            convert_type = "Binary",
            letter_box=False):
        self.target_shape = target_shape
        self.convert_type = convert_type
        self.letter_box = letter_box
        self.resize = resize
        self.crop = crop
        self.dtc = dtc
        self.mean = [0.485, 0.456, 0.406]
        self.std =  [0.229, 0.224, 0.225]

    def letterBox(self, src:np.array, size:list, interp:int=cv.INTER_AREA):
        ratio = np.minimum(size[0] / src.shape[0], size[1] / src.shape[1])
        insize_h = int(np.round(src.shape[0] * ratio))
        insize_w = int(np.round(src.shape[1] * ratio))
        pad_h = (size[0] - insize_h) / 2
        pad_w = (size[1] - insize_w) / 2

        top, bottom = int(np.round(pad_h - 0.1)), int(np.round(pad_h + 0.1))
        left, right = int(np.round(pad_w - 0.1)), int(np.round(pad_w + 0.1))

        dst = cv.resize(src, [insize_w, insize_h], fx=0, fy=0, interpolation=interp)
        dst = cv.copyMakeBorder(dst, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))

        return dst, (ratio, top, left, insize_h, insize_w, src.shape)
    def convert(self, image_path, save_path = None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像：{image_path}")

        if self.resize != None:
            image = cv2.resize(image, self.resize)
        after_letterbox, info = self.letterBox(image, self.target_shape, interp=cv.INTER_LINEAR)
        if self.letter_box:
            image = after_letterbox

        if self.crop !=None:
            h, w = image.shape[0], image.shape[1]
            y0 = (h - self.crop[0]) // 2
            x0 = (w - self.crop[1]) // 2
            image = image[y0 : y0 + self.crop[0], x0 : x0 + self.crop[1], :]  # hwc

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.convert_type == "YUV420SP":
            image_yuv420sp = yuv420sp(image, self.target_shape[0], self.target_shape[1]) # quant scs
            if save_path != None:
                with open(save_path + ".yuv420sp", "wb") as f:
                    f.write(image_yuv420sp.tobytes())
            return image_yuv420sp, info

        elif self.convert_type == "Binary":
            image = image.astype(np.float32) / 255.
            if self.dtc:
                image = (image - self.mean) / self.std
            image = np.transpose(image, axes=[2, 0, 1])
            image = image.astype(np.float32)
            if save_path != None:
                image.tofile(save_path + ".bin")
            return image, info
        else:
            print("convert_type err:", self.convert_type)
        return None, None


class Board_server():
    def __init__(self, boardIp, port, model_path, timeout = 3, need_load = True):
        self.boardIp = boardIp
        self.timeout = timeout
        self.base_url = "http://" + boardIp + ":" + port
        self.url = "http://" + boardIp + ":" + port + "/model"
        self.onnx_runner = None
        self.model_path = os.path.basename(model_path)
        if model_path.endswith("onnx"):
            self.onnx_runner = Onnx_Runner(model_path)
            print("creat onnx runner :", model_path)
        else:
            if need_load:
                self.loade_board(model_path)
                print("creat ", self.url)

    def cleanup(self):
        if self.model_path.endswith(".ortm") :
            self.unload(self.model_id)
            print("clean up: ", self.model_path, self.model_id)

    def __del__(self):
        if self.model_path.endswith(".ortm") :
            self.unload(self.model_id)
            print("clean up: ", self.model_path, self.model_id)

    def get_info(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()  # 检查请求是否成功
            response = response.json()
            print("current_info:", response)
            return response
        except requests.exceptions.RequestException as e:
            print("requests faile:", e)

    def get_model_info(self):
        try:
            response = requests.get(self.url + '/' + self.model_id)
            response.raise_for_status()  # 检查请求是否成功
            response = response.json()
            print("current_info:", response)
            return response
        except requests.exceptions.RequestException as e:
            print("requests faile:", e)

    def get_version(self):
        try:
            response = requests.get(self.base_url + '/version')
            response.raise_for_status()  # 检查请求是否成功
            response = response.json()
            print("response:", response)
            return response
        except requests.exceptions.RequestException as e:
            print("requests faile:", e)

    def get_device(self):
        try:
            response = requests.get(self.base_url + '/device')
            response.raise_for_status()  # 检查请求是否成功
            response = response.json()
            print("response:", response)
            return response
        except requests.exceptions.RequestException as e:
            print("requests faile:", e)

    def loade_board(self, model_path):
        filename = os.path.basename(model_path)
        file_obj = open(model_path, "rb")
        model_size = os.path.getsize(model_path)
        print('file_size',str(model_size))
        encoder = MultipartEncoder(
            fields={
                'file_size': (str(model_size), '', 'model/size'),
                'model': (filename, file_obj, 'model/octet-stream'),
            }
        )
        headers = {"file_size": str(model_size), 'Content-Type': encoder.content_type}
        try:
            response = requests.post(
                self.url,
                data=encoder,
                headers=headers,
                timeout=30
            )
            print(f"Response: {response.status_code} - {response.text}")
            res = json.loads(response.text)
            self.model_id = res['model_id']
        except Exception as e:
            print(f"Upload failed: {str(e)}")
        finally:
            file_obj.close()

    def unload(self, model_id):
        infer_url = self.url + '/' + model_id
        response = requests.delete(infer_url)
        # print(response.text)

    def init_npu(self):
        tmp_url = self.base_url + '/init'
        response = requests.post(tmp_url)
        print(f"init_npu, response: {response.json()}")

    def init_npu_with_profiling(self):
        infer_url = self.base_url + '/init/profiling'
        response = requests.post(infer_url)
        print(f"init_npu_with_profiling, response: {response.json()}")

    def init_npu_with_dump(self):
        infer_url = self.base_url + '/init/dump'
        headers = {"dump_path": "/data/LayerDump"}
        response = requests.post(infer_url, headers=headers)
        print(f"init_npu_with_dump, response: {response.json()}")

    def deinit_npu(self):
        infer_url = self.base_url + '/deinit'
        response = requests.post(infer_url)
        print(f"deinit_npu, response: {response.json()}")

    # infer with data
    def infer_once_with_tensor(self, data):
        #  data 要求为list
        if self.onnx_runner != None:
            return self.onnx_runner.process(data)
        infer_url = self.url + '/' + self.model_id
        if data is None:
            print("There is no data! Please check data!")
            return []
        if not isinstance(data, list):
            data = [data]
        input_nums = len(data)
        files = {}
        for input_num in range(input_nums):
            if isinstance(data[input_num], torch.Tensor):
                tensor_bytes = io.BytesIO(data[input_num].numpy().tobytes())
            elif isinstance(data[input_num], np.ndarray):
                tensor_bytes = io.BytesIO(data[input_num].tobytes())
            files["input." + str(input_num)] = tensor_bytes
        response = requests.put(infer_url, files=files)
        if response.status_code == 200:
            return response.content
        return []

    # infer with file
    def infer_once_with_file(self, data_path):
        if self.onnx_runner != None:
            data = np.fromfile(data_path, np.float32)
            return self.onnx_runner.process(data)
        infer_url = self.url + '/' + self.model_id
        if os.path.isdir(data_path):
            files = {}
            root, dirs, files in os.walk(data_path)
            file_nums = len(files)
            for file_num in range(file_nums):
                file_path = os.path.join(data_path, str(file_num) + ".bin")
                with open(file_path, 'rb') as f:
                    files["input_" + str(file_num)] = f
        else:
            infer_url = self.url + '/' + self.model_id
            with open(data_path, 'rb') as f:
                files = {'input.1': f}
                response = requests.put(infer_url, files=files)
                # print("put response length:", len(response.content))
                if response.status_code == 200:
                    arr = np.frombuffer(response.content, dtype=np.float32)
                    return arr
        response = requests.put(infer_url, files=files)
            # print("put response length:", len(response.content))
        if response.status_code == 200:
            arr = np.frombuffer(response.content, dtype=np.float32)
            return arr
        return []

    # infer for perf
    def infer_for_perf(self):
        # print('model_id:', self.model_id)
        infer_url = self.url + '/perf/' + self.model_id
        response = requests.put(infer_url)
        print(f"infer_for_perf, put response length: {len(response.content)}, json: {response.json()}")

    def infer_for_profiling(self):
        # print('model_id:', self.model_id)
        infer_url = self.url + '/profiling/' + self.model_id
        response = requests.put(infer_url)
        print(f"infer_for_profiling, put response length: {len(response.content)}, json: {response.json()}")
        csv_path = response.json().get("csv_path")[0]

        try:
            print(f"file_path:{csv_path}")
            headers = {"file_path": csv_path}
            response = requests.get(self.base_url + '/files/one_file_path', headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            response = response.text
            print("response:", response)
            return response
        except requests.exceptions.RequestException as e:
            print("requests fail:", e)

    def infer_for_dump(self, data_path):
        print(f'model_id:{self.model_id}, data_path:{data_path}')
        infer_url = self.url + '/dump/' + self.model_id
        with open(data_path, 'rb') as f:
            # print('model_id:', self.model_id)
            files = {'input.1': f}

            response = requests.put(infer_url, files=files)
            print(f"infer_for_dump, put response length: {len(response.content)}, json: {response.json()}")
            file_path = response.json().get("file_path")[0]

            try:
                print(f"file_path:{file_path}")
                headers = {"file_path": file_path}
                response = requests.get(self.base_url + '/files/one_file_path', headers=headers)
                response.raise_for_status()  # 检查请求是否成功

                response = requests.delete(self.base_url + '/files/one_file_path', headers=headers)
                response.raise_for_status()  # 检查请求是否成功

                # response = response.text
                # print("response:", response)
                return response
            except requests.exceptions.RequestException as e:
                print("requests fail:", e)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def save_target2txt(targets: np.ndarray, save_name: str):
    np.savetxt(str(save_name), targets, fmt="%g", delimiter=" ")

def xyxycn_to_nxywhc(boxes):
    boxes = np.asarray(boxes)
    if boxes.size:
        x1, y1, x2, y2, conf, n = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        boxes = np.column_stack((n, x, y, w, h, conf))

    return boxes

def nxywhc_to_normalization(boxes, image_hw):
    rh, rw = image_hw[0], image_hw[1]  # 获取高度、宽度
    if len(boxes) == 0:
        return boxes
    assert rw > 0 and rh > 0, "rw and rh must be positive"
    boxes = np.atleast_2d(boxes)
    boxes = np.asarray(boxes)
    n, x, y, w, h, conf = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
    x = x / rw
    y = y / rh
    w = w / rw
    h = h / rh

    new_boxes = np.column_stack((n, x, y, w, h, conf))
    return new_boxes

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou_torch(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    left   = max(box1[0], box2[0])
    top    = max(box1[1], box2[1])
    right  = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    cross  = max((right-left), 0) * max((bottom-top), 0)
    union  = area_box(box1) + area_box(box2) - cross
    iou = 0

    if cross != 0 and union != 0:
        iou = cross / union
    return iou

class Mod(object):
    def __init__(self, board_runner:Board_server, processer:DatasetCreater, process_type, conf_thresh:float=0.25, iou_thresh:float=0.5, scale = [], zp = []) -> None:
        self.__board_runner = board_runner
        self.__preprocesser = processer
        self.stride = np.array([8.0, 16.0, 32.0], dtype=np.float32)
        # yolov7 anchor
        self.anchor_v7 = np.array([
            [[  12,  16], [  19,  36], [  40,  28]],
            [[  36,  75], [  76,  55], [  72, 146]],
            [[ 142, 110], [ 192, 243], [ 459, 401]]
        ], dtype=np.float32)
        # yolov5 anchor
        self.anchor_v5 = np.array([
            [[  10, 13 ], [  16,  30 ], [  33,  23 ]],
            [[  30, 61 ], [  62,  45 ], [  59, 119 ]],
            [[ 116, 90 ], [ 156, 198 ], [ 373, 326 ]]
        ], dtype=np.float32)
        # yolov5 需要在后处理中做sigmoid
        self.need_sigmoid = board_runner.model_path.lower().find("yolov5") != -1
        self.in_shape = [640, 640]
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        if process_type == "yolov3":
            self.forward = self.forward_yolov3_no_postprocess
        elif process_type == "yolov5_cve":
            self.forward = self.forward_cve_postprocess
        elif process_type == "yolov7":
            self.forward = self.forward_yolov5_7_no_postprocess
        elif process_type == "yolov8":
            self.forward = self.forward_yolov8_10_11
        self.anchor = self.anchor_v7
        if board_runner.model_path.lower().find("yolov5") != -1:
            self.anchor = self.anchor_v5
        self.scale = scale
        self.zp = zp
        self.__cls = 80

    def cropByBound(self, pred, size):
        if pred.size != 0:
            pred[..., 0] = np.maximum(pred[..., 0], 0)
            pred[..., 1] = np.maximum(pred[..., 1], 0)
            pred[..., 2] = np.minimum(pred[..., 2], size[1])
            pred[..., 3] = np.minimum(pred[..., 3], size[0])
        return pred
    
    def forward_cve_postprocess(self, image_path):
        after_preprocess, info = self.__preprocesser.convert(image_path)
        pred = self.__board_runner.infer_once_with_tensor(after_preprocess)
        if len(pred) == 2404 or len(pred) == 2416:
            pred = np.frombuffer(pred, dtype=np.float32).copy()
            num_targets = int(pred[0]) # 有效框个数
            box_data = pred.reshape([-1])[-600:].reshape(6, -1) # [6, 100]
            box_data = box_data.transpose(1,0) # [100, 6]
            outputs = box_data[:num_targets, :] # 有效框 xyxycn
            outputs = self.scale_2_src(outputs, info)
            outputs = self.cropByBound(outputs, info[5])
            outputs = xyxycn_to_nxywhc(outputs)
            outputs = nxywhc_to_normalization(outputs, info[5]) # 坐标归一化
            return outputs
        return []
    
    def decode_yolov3(self, out, stride, anchor_grid, seg=False, need_sigmoid=False):
        # output [[1,255,20,20], [1,255,40,40]]
        auchor=[[[81,82],[135,169],[344,319]], [[10,14],[23,27],[37,58]]]
        for i in range(2):
            out[i] = out[i].reshape(1,3,85,i*20+20, i*20+20).transpose(0,1,3,4,2)
            out[i] = sigmoid(out[i])
            out[i][...,0:4] *= 2
            grid = make_grid(i*20+20, i*20+20) - 0.5
            out[i][...,0:2] += grid
            out[i][...,0:2] *= 32/(i+1)

            out[i][...,2:4] *= out[i][...,2:4]
            for j in range(3):
                out[i][0][j][...,2:4] *= auchor[i][j]
            out[i] = out[i].reshape(1, -1, 85)
        return np.concatenate(out, 1)
    
    # 输出为[[1,255,20,20], [1,255,40,40]] 且没有做sigmoid
    def forward_yolov3_no_postprocess(self, image_path):
        after_preprocess, info = self.__preprocesser.convert(image_path)
        pred = self.__board_runner.infer_once_with_tensor(after_preprocess)
        if len(pred) == 2:
            outputs = pred
        else:
            if len(pred) == 510000:
                pred = np.frombuffer(pred, dtype=np.int8).copy().astype(np.float32)
                outputs = np.array(pred).reshape([-1])
                outputs = [outputs[:102000], outputs[102000:]]
                for i in range(2):
                    outputs[i] = (outputs[i] - self.zp[i]) * self.scale[i]
            elif len(pred) == 510000*2:
                pred = np.frombuffer(pred, dtype=np.float16).copy().astype(np.float32)
                outputs = np.array(pred).reshape([-1])
                outputs = [outputs[:102000], outputs[102000:]]
            elif len(pred) != 510000 * 4:
                print("infer err")
                return None
            else:
                pred = np.frombuffer(pred, dtype=np.float32)
                outputs = np.array(pred).reshape([-1])
                outputs = [outputs[:102000], outputs[102000:]]
        outputs = [out.reshape([1, 3, 85, -1]) for out in outputs]
        outputs = self.decode_yolov3(outputs, self.stride, self.anchor, seg=False, need_sigmoid=self.need_sigmoid)
        outputs = self.non_max_suppression(outputs,self.conf_thresh, self.iou_thresh)[0]
        outputs = self.scale_2_src(outputs, info)
        outputs = self.cropByBound(outputs, info[5])
        outputs = xyxycn_to_nxywhc(outputs)
        outputs = nxywhc_to_normalization(outputs, info[5]) # 坐标归一化
        return outputs
    
    def forward_yolov5_7_no_postprocess(self, image_path):
        after_preprocess, info = self.__preprocesser.convert(image_path)
        pred = self.__board_runner.infer_once_with_tensor(after_preprocess)
        if len(pred) == 3:
            outputs = pred
        else:
            if len(pred) == 2142000:
                pred = np.frombuffer(pred, dtype=np.int8).copy().astype(np.float32)
                outputs = np.array(pred).reshape([-1])
                outputs = [outputs[:6400*255], outputs[6400*255:8000*255], outputs[-400*255:]]
                for i in range(3):
                    outputs[i] = (outputs[i] - self.zp[i]) * self.scale[i]
            elif len(pred) == 4284000:
                pred = np.frombuffer(pred, dtype=np.float16).copy().astype(np.float32)
                outputs = np.array(pred).reshape([-1])
                outputs = [outputs[:6400*255], outputs[6400*255:8000*255], outputs[-400*255:]]
            elif len(pred) != 2142000 * 4:
                print("infer err")
                return None
            else:
                pred = np.frombuffer(pred, dtype=np.float32)
                outputs = np.array(pred).reshape([-1])
                outputs = [outputs[:6400*255], outputs[6400*255:8000*255], outputs[-400*255:]]
        outputs = [out.reshape([1, 3, 85, -1]) for out in outputs]
        outputs = [out.transpose(0,1,3,2) for out in outputs]
        outputs = [outputs[0].reshape([1, 3, 80, 80, 85]), outputs[1].reshape([1, 3, 40, 40, 85]), outputs[2].reshape([1, 3, 20, 20, 85])]
        outputs = self.decode_yolov5_7(outputs, self.stride, self.anchor, seg=False, need_sigmoid=self.need_sigmoid)
        outputs = self.non_max_suppression(outputs,self.conf_thresh, self.iou_thresh)[0]
        outputs = self.scale_2_src(outputs, info)
        outputs = self.cropByBound(outputs, info[5])
        outputs = xyxycn_to_nxywhc(outputs)
        outputs = nxywhc_to_normalization(outputs, info[5]) # 坐标归一化
        return outputs

    def NMS(self, boxes, iou_thresh):
        boxes = sorted(boxes, key=lambda x:x[4], reverse=True)
        remove_flags = [False] * len(boxes)
        keep_boxes = []

        for i, ibox in enumerate(boxes):
            if remove_flags[i]:
                continue

            keep_boxes.append(ibox)
            for j in range(i + 1, len(boxes)):
                if remove_flags[j]:
                    continue

                jbox = boxes[j]
                # if(ibox[5] != jbox[5]):
                #     continue
                if iou(ibox, jbox) > iou_thresh:
                    remove_flags[j] = True

        return np.array(keep_boxes)
    
    def forward_yolov8_10_11(self, image_path):
        after_preprocess, info = self.__preprocesser.convert(image_path)
        ratio, top, left, insize_h, insize_w, src_shape = info
        pred = self.__board_runner.infer_once_with_tensor(after_preprocess)
        
        preds=[]
        if len(pred) == 6:
            preds = pred
        else:
            if len(pred) == 1209600*2:
                pred = np.frombuffer(pred, dtype=np.float16).copy().astype(np.float32)
            elif len(pred) == 1209600*4:
                pred = np.frombuffer(pred, dtype=np.float32).copy()
            else:
                print("output formate err")
                return None
            x1, x2, x3, x4, x5 = 64*80*80, 144*80*80, 144*80*80+64*40*40, 144*80*80+144*40*40, 144*80*80+144*40*40+64*20*20
            preds = [pred[:x1].reshape([1,64,80,80]),
                        pred[x1:x2].reshape([1,80,80,80]),
                        pred[x2:x3].reshape([1,64,40,40]),
                        pred[x3:x4].reshape([1,80,40,40]),
                        pred[x4:x5].reshape([1,64,20,20]),
                        pred[x5:].reshape([1,80,20,20])]
        boxes8 = self.__post_process_wodfl(preds[0], preds[1], 8)
        boxes16 = self.__post_process_wodfl(preds[2], preds[3], 16)
        boxes32 = self.__post_process_wodfl(preds[4], preds[5], 32)
        boxes = boxes8 + boxes16 + boxes32
        if len(boxes) == 0:
            return boxes
        outputs = self.NMS(boxes, self.iou_thresh)
        outputs = self.scale_2_src(outputs, info)
        outputs = self.cropByBound(outputs, info[5])
        outputs = xyxycn_to_nxywhc(outputs)
        outputs = nxywhc_to_normalization(outputs, info[5])
        return outputs


    def __post_process_wodfl(self, pred, cls_pred, stride):
        def softmax(x, axis=-1, epsilon=1e-15):
            """数值稳定的Softmax函数，支持多维输入"""
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max) + epsilon  # 防止全零
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        boxes = []
        reg_max = 16
        _, _, fh, fw = pred.shape
        cls_pred = np.transpose(cls_pred[0], (1, 2, 0)).reshape(-1, self.__cls)
        pred = np.transpose(pred[0], (1, 2, 0)).reshape(-1, 4*reg_max)
        ind = np.arange(reg_max)
        for i in range(len(cls_pred)):
            item = cls_pred[i]
            label = item[:].argmax()
            confidence = sigmoid(item[label])
            if confidence < self.conf_thresh:
                continue
            box = pred[i]
            l, t, r, b =  np.split(box, 4)

            l = np.sum(softmax(l)*ind)
            t = np.sum(softmax(t)*ind)
            r = np.sum(softmax(r)*ind)
            b = np.sum(softmax(b)*ind)

            hi = i // fw
            wi = i - hi * fw

            left = ((wi + 0.5) - l)*stride
            top = ((hi + 0.5) - t)*stride
            right = ((wi + 0.5) + r)*stride
            bottom = ((hi + 0.5) + b)*stride

            boxes.append([left, top, right, bottom, confidence, label])

        return boxes

    def scale_2_src(self, pred, info):
        if pred.size != 0:
            if len(pred.shape) > 2:
                pred = pred.reshape([-1, pred.shape[-1]])
            pred[..., [0, 2]] -= info[2]
            pred[..., [1, 3]] -= info[1]
            pred[..., :4] /= info[0]

        return pred


    # 解码yolov5/7 三个检测头的结果
    # [1,3,80,80,85] [1,3,40,40,85] [1,3,20,20,85]
    def decode_yolov5_7(self, out, stride, anchor_grid, seg=False, need_sigmoid=False):
        nc = 80
        boxc = 5
        seg_part = 0
        if seg == True:
            seg_part = 32
        stride = np.array(stride)
        anchor_grid = np.array(anchor_grid).reshape(-1, 3, 2)[:, None, :, None, None, :]
        z = []
        for i in range(3):
            out[i] = out[i].reshape(out[i].shape[0], 3, out[i].shape[3], out[i].shape[2], nc + boxc + seg_part)

            grid = make_grid(out[i].shape[3], out[i].shape[2])
            y = out[i]
            if need_sigmoid:
                y[:, :, :, :, :85] = sigmoid(y[:, :, :, :, :85])
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2)**2 * anchor_grid[i]  # wh
            z.append(y.reshape(out[i].shape[0], -1, nc + boxc + seg_part))
        return np.concatenate(z, 1)

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
    ):
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        # 添加NumPy数组支持
        numpy_input = False
        if isinstance(prediction, np.ndarray):
            numpy_input = True
            prediction = torch.from_numpy(prediction)  # 转换为PyTorch张量
        
        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = "mps" in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                iou = box_iou_torch(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded

        # 如果需要返回NumPy格式
        if numpy_input:
            output = [tensor.numpy() for tensor in output]

        return output

# 分类网预处理
class Preprocess_val_bin(object):
    def __init__(self, model_name, convert_type):
        dataset_256 = ["resnet.*18", "resnet34", "resnet50", "resnet101", \
               "vgg16", "vgg19", "densenet121", \
               "mobilenet.*v3", "mobilenet.*v2", \
               "alexnet-13"]
        dataset_299 = ["inception_v3"]
        model_name = os.path.basename(model_name)
        self.resize = 256
        self.crop = 224
        if any(re.search(target, model_name) for target in dataset_299):
            self.resize = 299
            self.crop = 299
            self.keep_aspect_ratio = True
        else:
            self.keep_aspect_ratio = any(re.search(target, model_name) for target in dataset_256)
        self.yuv420sp = convert_type == "YUV420SP" and model_name[-4:] == "ortm"
        print(model_name, "resize:", self.resize, "crop:", self.crop, "keep_aspect_ratio:", self.keep_aspect_ratio, "yuv420sp:", self.yuv420sp)

    def preprocess(self, image_path:str):
        img = cv2.imread(image_path)
        if len(img.shape) < 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.keep_aspect_ratio:
            height, width = img.shape[:2]
            scale = self.resize / min(height, width)
            new_size = (math.ceil(width * scale), math.ceil(height * scale))
            img = cv2.resize(img, new_size)
        else:
            img = cv2.resize(img, (self.resize, self.resize))
        h, w = img.shape[0], img.shape[1]
        y0 = (h - self.crop) // 2
        x0 = (w - self.crop) // 2
        img = img[y0 : y0 + self.crop, x0 : x0 + self.crop, :]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.yuv420sp:
            return yuv420sp(img, self.crop, self.crop) #
        img = img / 255.
        mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        scale = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / scale
        img = np.transpose(img, axes=[2, 0, 1])
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

