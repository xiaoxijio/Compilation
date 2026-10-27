# mobilenetv2 Sample
- 支持resnet18-13、resnet50、mobilenetv2、vgg16-12

## ONNX 模型来源
[mobilenetv2-7](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx)

## Acc精度评估
### 数据集准备
- 06_evaluate.sh在执行时，会自动检查、下载并准备评估数据集，如需手动部署可以参考以下命令
- 执行解压命令
```bash
cd samples/data/evaluate_dataset
mkdir -p ILSVRC2012/ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012/ILSVRC2012_img_val/
```
- 数据集结构
```plaintext
└── samples/data/evaluate_dataset/
    └──ILSVRC2012/
       ├──ILSVRC2012_img_val/
       └──ILSVRC2012_img_val.txt
```

### 执行评估
```bash
./06_evaluate.sh
```