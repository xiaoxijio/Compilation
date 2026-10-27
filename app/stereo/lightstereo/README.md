# lightstereo Sample

## 项目来源
[GitHub 仓库](https://github.com/XiandaGuo/OpenStereo)

## PyTorch 模型来源
[模型下载链接](https://drive.google.com/drive/folders/1qG02SQLLaXT35vVTAS9UWvCWfB6RPgEu)

## 公版模型和魔改版本切换
修改参数详见`samples/script/platform_config.sh`
```bash
ENABLE_OPENSOURCE_ONNX=0 # 选择开源lightstereo/魔改lightstereo ONNX, 1:开源，0: 魔改
```

## 模型精度评估

参考: `samples/app/stereo/lightstereo/evaluate/README.md`

## 版本支持
目前该模型仅支持V200上执行