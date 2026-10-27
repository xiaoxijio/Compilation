## 一、环境配置
### 1. cuda版本设置
统一选择cuda11.3

在~/.bashrc中添加以下配置：
```shell
    # 设置CUDA版本
    cuda_version="11.3"
    # 设置CUDA安装路径
    cuda_home="/usr/local/cuda-${cuda_version}"
    # 更新PATH变量
    export PATH=/usr/local/cuda-${cuda_version}/bin:$PATH
    # 更新LD_LIBRARY_PATH变量
    export LD_LIBRARY_PATH=/usr/local/cuda-${cuda_version}/lib64:$LD_LIBRARY_PATH
    # 更新CUDA_HOME变量
    export CUDA_HOME=/usr/local/cuda-${cuda_version}
```

每次运行前,执行 source ~/.bashrc
### 2. 创建mmdet3d环境 
* 如果已有本sample中的mmdet3d conda环境, 可不用创建新的conda环境, 按照如下步骤重定位mmdet3d即可
```shell
    conda activate mmdet3d
    # 卸载原有mmdet3d
    pip uninstall mmdet3d
    # 在当前evaluate路径下执行
    pip install -v -e .
```

* 创建纯净环境
```shell
    conda create -n mmdet3d python=3.8 
    conda activate mmdet3d
    conda install pip
    # 使用 python -m pip install 确保pip和python是配套的
    python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113  

    python -m pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html  
    python -m pip install mmdet==2.25.0  
    python -m pip install mmsegmentation==0.26.0  

    # 下载 mmdet3d-1.0.0rc6 版本  url:  https://files.pythonhosted.org/packages/7c/38/5abed519e2c644bd6f447e73128d22ecfe2d55e3672445065ae8ee540b76/mmdet3d-1.0.0rc6.tar.gz
    wget https://files.pythonhosted.org/packages/7c/38/5abed519e2c644bd6f447e73128d22ecfe2d55e3672445065ae8ee540b76/mmdet3d-1.0.0rc6.tar.gz
    tar -zxvf mmdet3d-1.0.0rc6.tar.gz
    mv mmdet3d-1.0.0rc6 mmdet3d
    cd mmdet3d
    python -m pip install setuptools==69.2.0   ## 放前面，不然版本可能导致后续安装报错
    python -m pip install -v -e.  
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'git+https://github.com/facebookresearch/detectron2.git'
    python -m pip install seaborn==0.13.2
    python -m pip install numpy==1.22.4  
    python -m pip install yapf==0.40.0  
    python -m pip install IPython  
    python -m pip install fvcore==0.1.5.post20221221
    python -m pip install ipdb==0.13.13  
```

## 二、nuScenes测试集准备
nuScenes 抽取自全量数据集 `v1.0-trainval` 的`val`

组织结构如下：
```
├── samples/app/vision_bev/mod/common/evaluate/data/
    ├── maps
    ├── samples
    ├── sweeps
    ├── v1.0-trainval
    ├── can_bus
    ├── bevdetv3-nuscenes_gt_database
    ├── nuscenes_infos_temporal_val.pkl
    └── bevdetv3-nuscenes_infos_val.pkl
```

## 三、评估
### 1. 生成预处理数据文件
```shell
# fastbev
python save_preprocessed_imgs.py --config fastbev --save_path ./preprocessed_imgs/
# bevdet
python save_preprocessed_imgs.py --config bevdet --save_path ./preprocessed_imgs/

# 注意：此处 --save_path 需要与  06_evalutae.sh 中的 --data_path 保持一致
```

nuScenes数据测试集共6019帧，保存的预处理图像bin文件形式为：
```shell
preprocessed_imgs/${SAMPLE_TOKEN}/${SAMPLE_TOKEN}_6x3x256x704.bin
....
# 共6019个bin
```

### 2. 批量生成板端输出
在samples/script/platform_config.sh中对板端IP以及Port进行配置(此处示例IP:10.16.81.27,Port:1234)

进入到IP为10.16.81.27板端环境，在mount路径下执行命令：
``` shell
./avp_server -p 1234 (1234为监听端口号)
```
板端出现如下打印即成功建立连接
``` shell
Listen on port 1234
``` 
在对应模型目录下 `fastbev++` / `bevdet` 下执行

```shell
./06_evalutae.sh

```

### 3. 板端输出精度验证
```shell
# fastbev
python board_sample_inference.py --config configs/fastbev/fastbev-r50-cbgs.py --board_out_path ./preprocessed_imgs/
# bevdet
python board_sample_inference.py --config configs/bevdet/bevdet-r50-cbgs.py --board_out_path ./preprocessed_imgs/

```
