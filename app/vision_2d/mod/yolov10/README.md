# YOLOv10 Sample

## 项目来源
[GitHub 仓库](https://github.com/THU-MIG/yolov10)

## PyTorch 模型来源
[yolov10n.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt)
[yolov10s.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt)
[yolov10m.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt)
[yolov10b.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt)
[yolov10l.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt)
[yolov10x.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt)

## ONNX 模型转换方式(以yolov10s模型为例)
```bash
# 将官方的pytorch模型转为onnx模型
yolo export model=yolov10s.pt format=onnx opset=13
# 保留yolov10主干网络
python modify_onnx.py -m yolov10s.onnx -s yolov10s_modify.onnx -o \
    /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0 \
    /model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0 \
    /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0 \
    /model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0 \
    /model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0 \
    /model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0

```

## map精度评估
- 参照 app/vision_2d/mod/yolov5/README.md