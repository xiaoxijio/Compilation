# flashocc Sample

## 项目来源
[GitHub 仓库](https://github.com/Yzichen/FlashOCC)

## PyTorch 模型来源
[模型下载链接](https://drive.google.com/file/d/14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ/view?usp=sharing)


## 公版模型和魔改版本切换
修改参数详见`samples/script/platform_config.sh`
```bash
ENABLE_OPENSOURCE_ONNX=0 # 选择开源flashocc/魔改flashocc ONNX, 1:开源，0: 魔改
```

## 环境准备
在运行本sample前需要准备好conda环境，参考[环境配置](../evaluate/README.md)

## 模型精度评估

### 1. 板端打开avp_server
在samples/script/platform_config.sh中对板端IP以及Port进行配置(此处示例IP:10.16.81.27,Port:1234)

进入到IP为10.16.81.27板端环境，在mount路径下执行命令：
``` shell
./avp_server -p 1234 (1234为监听端口号)
```
板端出现如下打印即成功建立连接
``` shell
Listen on port 1234
``` 

### 2. 修改参数

./06_evalutae.sh:7 

ENABLE_OPENSOURCE_ONNX ：1为公版模型，0为魔改模型

### 3. 执行评估
```shell
./06_evalutae.sh
```

## 模型可视化

参考: `samples/app/vision_occ/visualization/README.md`

## 版本支持
目前该模型仅支持V200上执行
