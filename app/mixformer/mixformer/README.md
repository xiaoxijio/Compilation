# MixFormerV2 Sample

## 项目来源

[Github 仓库](https://github.com/MCG-NJU/MixFormerV2)

## PyTorch 模型来源

[模型下载链接](https://drive.google.com/drive/folders/1soQMZyvIcY7YrYrGdk6MCstTPlMXNd30?usp=sharing) 

## ONNX 模型转换方式


## 环境准备
注意：在执行本sample前需先激活所需python环境

如需新建环境：参考如下：
```bash
conda create -n mixformer2  python=3.10
conda activate mixformer2
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install PyYAML
pip install cython
pip install opencv-python
pip install pandas
conda install -y tqdm
pip install pycocotools
pip install jpeg4py
pip install tikzplotlib
pip install lmdb
pip install scipy
pip install onnxruntime
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
