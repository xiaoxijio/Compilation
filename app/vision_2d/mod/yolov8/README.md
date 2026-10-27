# YOLOv8 Sample

## PyTorch 模型来源
[yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)

## ONNX 模型转换方式(以yolov8n模型为例)
```bash
# 将官方的pytorch模型转为onnx模型
yolo export model=yolov8n.pt format=onnx opset=13

python modify_onnx.py -m yolov8n.onnx -s yolov8n_modify.onnx -o \
    /model.22/cv2.0/cv2.0.2/Conv_output_0 \
    /model.22/cv3.0/cv3.0.2/Conv_output_0 \
    /model.22/cv2.1/cv2.1.2/Conv_output_0 \
    /model.22/cv3.1/cv3.1.2/Conv_output_0 \
    /model.22/cv2.2/cv2.2.2/Conv_output_0 \
    /model.22/cv3.2/cv3.2.2/Conv_output_0 

```

## map精度评估
- 参照 app/vision_2d/mod/yolov5/README.md