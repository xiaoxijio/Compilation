# bevdet Sample

## 项目来源
[GitHub 仓库](https://github.com/HuangJunJie2017/BEVDet)

## PyTorch 模型来源
[模型下载链接](https://pan.baidu.com/share/init?surl=237QyV18zvRJ1pU3YzRItw&pwd=npe1)

## ONNX 模型转换方式
使用以下命令：
```bash
python export_pt2onnx.py bevdet.pt bevdet.onnx
```

## 公版模型和魔改版本切换
修改参数详见`samples/script/platform_config.sh`
```bash
ENABLE_OPENSOURCE_ONNX=0 # 选择开源bevdet/魔改bevdet ONNX, 1:开源，0: 魔改
```

## 环境配置
在运行本sample前需要准备好conda环境，参考[环境配置](../common/evaluate/README.md)

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
### 2. 执行评估
```shell
./06_evalutae.sh
```

## 版本支持
目前该模型仅支持V200上执行
