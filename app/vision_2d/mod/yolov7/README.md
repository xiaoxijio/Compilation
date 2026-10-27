# YOLOv7 Sample
支持yolov7, yolov7x, yolov7-tiny

## 项目来源  
[GitHub 仓库](https://github.com/WongKinYiu/yolov7.git)  

## PyTorch 模型来源  
- [yolov7-tiny.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt)  
- [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)  
- [yolov7x.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) 

## ONNX 模型转换方式(以yolov7模型为例)
```bash
# 克隆YOLOv7官方仓库
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7/

# 将PyTorch模型转换为ONNX格式
python export.py --weights yolov7.pt

# yolov7提取主干网络
python sample/modify_onnx.py -m yolov7.onnx -s yolov7_modify.onnx --need_sigmoid \
    -o /model.105/m.0/Conv_output_0 /model.105/m.1/Conv_output_0 /model.105/m.2/Conv_output_0

# yolov7-tiny提取主干网络
python sample/modify_onnx.py -m yolov7-tiny.onnx -s yolov7-tiny_modify.onnx --need_sigmoid \
    -o /model.77/m.0/Conv_output_0 /model.77/m.1/Conv_output_0 /model.77/m.2/Conv_output_0
```

## map精度评估
- 参照 app/vision_2d/mod/yolov5/README.md