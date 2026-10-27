# LSS Sample

## 项目来源

[Github 仓库](https://github.com/nv-tlabs/lift-splat-shoot)

## PyTorch 模型来源

[模型下载链接](https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing)

## ONNX 模型转换方式
参考[官方文档](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/README.md)


## 公版模型和魔改版本切换
修改参数详见`samples/script/platform_config.sh`
```bash
ENABLE_OPENSOURCE_ONNX=0 # 选择开源lss/魔改lss ONNX, 1:开源，0: 魔改
```

## 环境准备
在运行本sample前需要准备conda环境，如已有[bev系列环境](../../vision_bev/mod/common/evaluate/README.md)，可直接使用

如需新建环境：参考如下：
```bash
    conda create -n lss python=3.10
    conda activate lss
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install numpy==1.26.4
    pip install onnx 
    pip install onnxruntime-gpu==1.16.3 
    # 如果跑得慢, 可以换成onnxruntime-gpu==1.18.1
    pip install onnxsim==0.4.36 
    pip install onnxoptimizer==0.3.13 
    pip install onnx-tool==0.9.0
    pip install onnxconverter-common==1.15.0
    pip install nuscenes-devkit==1.1.11 tensorboardX==2.6.4 efficientnet_pytorch==0.7.0
    pip install pyquaternion==0.9.9
    pip install opencv-python==4.11.0.86
    pip install pillow==11.3.0
    pip install  tqdm==4.67.1
    pip install requests_toolbelt

```
## 精度评估


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
