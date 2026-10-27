# YOLOv5 Sample
支持yolov5n, yolov5s, yolov5m, yolov5l, yolov5x

## 项目来源
[GitHub 仓库](https://github.com/ultralytics/yolov5)

## PyTorch 模型来源
[yolov5n.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)
[yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
[yolov5m.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)
[yolov5l.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)
[yolov5x.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)

## 模型转换 （以yolov5s为例）-- 前处理请参考samples根目录下README.md
### 使用cve后处理
```bash
# clone YOLOV5官方仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
# 将官方的pytorch模型转为onnx模型
python export.py --weights ./yolov5s.pt --include onnx
# 提取主干网络
cd sample/script
python modify_onnx.py -m yolov5s.onnx -s yolov5s_modify.onnx -o /model.24/m.0/Conv_output_0 /model.24/m.1/Conv_output_0 /model.24/m.2/Conv_output_0
```
## map精度评估
### 数据集准备
- 06_evaluate.sh在执行时，会自动检查、下载并准备评估数据集，如需手动部署可以参考以下命令
- 执行解压命令
```bash
mkdir -p data/evaluate_dataset/COCO2017
unzip annotations_trainval2017.zip -d data/evaluate_dataset/COCO2017
unzip val2017.zip -d data/evaluate_dataset/COCO2017/
```
- 数据集结构
```plaintext
└── samples/data/evaluate_dataset/
    └──COCO2017/
       ├──annotations/
       └──val2017/
```

### 执行评估
```bash
./06_evaluate.sh
```