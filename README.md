# samples


## 一、 目录说明
# samples 目录说明

本目录 (`samples`) 用于存放各类 API 和模型的使用示例代码，旨在提供快速上手和参考。以下是目录结构和内容的详细说明。

---

## 目录结构概览
```plaintext
samples
├── api_tutorial                      # 用于存放各类 API 的使用示例代码
│   ├── async_forward                    # 异步执行和同步执行api用例
│   ├── binary_input                     # 读取二进制输入文件api用例
│   ├── custom_abs                       # 自定义算子api用例
│   ├── ortm_run_model                   # 通用api用例(静态库)
│   ├── sample_shared_lib                # 通用api用例(共享库)
│   ├── stride_align                     # stride对齐api用例
│   └── yuv420sp_input                   # 读取yuv420sp输入数据api用例
├── app                                # 用于存放各类模型应用
├── avp_server                         # 用于存放板端服务工具
├── build.sh                           # 编译脚本
├── cmake                              # 存放cmake编译文件
├── CMakeLists.txt                     # camke
├── data                               # 存放模型和数据集
├── README.md                          # 本文件
├── sample_utils                       # 通用代码文件
├── script                             # 脚本文件
└── third_party                        # 第三方库
```

## 二、 sample执行过程（以yolov5为例）

### step_0：环境准备
```
pip install virtualenv -i https://mirrors.aliyun.com/pypi/simple
virtualenv testenv
source testenv/bin/activate
pip install -r ./script/requirements.txt -i https://mirrors.aliyun.com/pypi/simple

```
### step_1：配置config（按照实际需求填写）
路径：script/platform_config.sh

- NPU_TOOLCHAIN_PATH：# npu_toolchain发布包根目录

- TARGET: 执行目标平台, 支持的列表：[LQ560PlusV100/LQ560MiniV110/LQ560PlusV200/LQ560ProV200等] (所有支持的目标平台具体见LQ560V200 NPU工具使用指南)

- PLATFORM：编译及运行的平台, 支持的类型：[simulator/board]

- OMG: omg所在的绝对路径

- BOARD_IP: 板端环境IP地址

- PORT：avp_server端口号(可在执行时手动配置，详见step_9)

### step_2：进入yolov5 sample所在路径
``` shell
cd app/vision_2d/mod/yolov5
```

### step_3：准备onnx和dataset
``` shell
./00_init.sh
```

### step_4：检查onnx能否转成ortm (此版本暂不支持)
``` shell
./01_check.sh
```

### step_5：量化数据集预处理
``` shell
./02_preprocess.sh
```

### step_6：omg转onnx到ortm
``` shell
./03_omg.sh
```

### step_7：编译可执行程序（内容：模型推理+后处理+可视化等）
``` shell
./04_build.sh
```

### step_8：执行可执行程序
``` shell
./05_inference.sh
```


### step_9：精度评估
#### 准备工作
#### 1.下载模型mAP所需测试数据集
- sample在执行06_evaluate.sh时，会自动检查、下载并准备需要的评估数据集，如需手动部署可以参考以下流程
- 建议下载路径：samples/data/evaluate_dataset/

- 链接中共有五个测试数据集文件夹（VOC2017,ILSVRC2012,COCO2017,nuscenes,lightstereo）, 一个LLM模型权重文件夹（model）

- 烦请根据所需sample下载指定的数据集
    - COCO2017 (包含app/vision_2d/mod/yolov5, app/vision_2d/mod/yolov7, app/vision_2d/mod/yolov8, app/vision_2d/mod/yolov10)

    - ILSVRC2012 (包含app/vision_2d/classification/mobilenetv2，app/vision_2d/classification/resnet50，app/vision_2d/classification/resnet152)

    - nuscenes (包含app/vision_bev/mod/bevdet，app/vision_bev/mod/fastbev++, app/vision_occ/flashocc)

    - lightstereo (包含app/stereo/lightstereo)

    - model (包含app/llm/qwen, app/llm/tinyllama)

#### 目录结构
```plaintext
├── samples/data/evaluate_dataset/
│   ├──COCO2017/
│      ├──annotations/
│      └──val2017/
│   ├──ILSVRC2012/
│      ├──ILSVRC2012_img_val/
│      └──ILSVRC2012_img_val.txt
└── ├──nuscenes/
       ├── maps
       ├── samples
       ├── sweeps
       ├── v1.0-trainval
       ├── can_bus
       ├── bevdetv3-nuscenes_gt_database
       ├── nuscenes_infos_temporal_val.pkl
       └── bevdetv3-nuscenes_infos_val.pkl
```

#### 2.测试数据集前处理
- fastbev++\bevdet使用nuScenes测试集进行mAP评估,需要将数据集软连接到相关路径，具体执行操作说明详见samples/app/vision_bev/mod/common/evaluate/README.md

- 检测网使用COCO20217测试集进行mAP评估，需要将数据通过python转换成.txt文件，具体执行操作说明可参考samples/app/vision_2d/mod/yolov5/README.md

- 分类网使用ILSVRC2012测试集进行mAP评估，数据集解压后即可供mAP使用，具体执行操作说明可参考samples/app/vision_2d/mod/classification/mobilenetv2/README.md

#### 3.板端服务工具及环境的准备
- 下载sdkv200发布包，解压后执行
``` shell
    cd sdk/
    ./build.sh
```
- 进入sdk中npu编译生成文件所在路径
``` shell
    cd ./build/main/component/media/npu/sample
```
- 该路径下会生成一个可执行文件
``` shell
    avp_server
```
- 将sdk/build/main/component/media/npu 执行mount操作到板端
- 此时，板端服务工具及mount路径都已准备就绪

#### 执行
- 在samples/script/platform_config.sh中对板端IP以及Port进行配置(此处示例IP:10.16.81.27,Port:1234)

- 进入到IP为10.16.81.27板端环境，在mount路径下执行命令：
``` shell
./avp_server -p 1234 (1234为监听端口号)
```
板端出现如下打印即成功建立连接
``` shell
Listen on port 1234
```

进入到app/vision_2d/mod/yolov5目录下，执行
``` shell
./06_evaluate.sh
```
#### 示例结果
服务端执行06_evaluate.sh后出现如下日志打印，即代表执行成功：
``` shell
load response: {
    "model_id": "bbbbd22b371606325039398903a0372c"
}
load model success id bbbbd22b371606325039398903a0372c
......
......
creat  http://xx.xx.xx.xx:1234/model
clean up:  yolov5s.ortm f7e8d295bb5aa419bab4299556c5f328
Loading data from  /xxx/xxx/samples/script/../data/evaluate_dataset/COCO2017/val2017/ tmp_res/pred/
Save annotation to tmp_res/result.json
......
......
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.244
......
......
```

## 三、自动生成sample工程脚本指南
需要预先准备好onnx的模型文件，建议放在samples/data/model目录下
``` shell
cd samples/script
python generate_custom_sample.py -m [onnx具体路径] -i [input_cfg.yml具体路径]
```
- -m: 指定onnx文件，生成的sample工程目录存放在samples/app/custom下[必选]
- -i: 指定input_cfg.yml文件，放到sample目录下；如果未指定，则会生成一个默认的yml文件，并且需要自行配置quant_dataset路径,calib_type, input_type及preprocess配置（配置可参考《LQ560V200 NPU工具使用指南.docx》章节2.3.4）[可选]
- 在生成的sample工程目录中，02_preprocess.sh会解析onnx模型，随机生成数个输入文件，作为sample输入，可以修改shell脚本，实现数据前处理，也可以在执行文件时，使用-i将输入文件所在的文件夹传入。

注意：通过该脚本生成的sample工程会自动更新samples/CMakeLists.txt编译文件。

## 四、windows编译执行说明（以api_tutorial\binary_input为例）
### 模型生成方式
- 参考《npu 工具使用指南》章节三生成samples中的模型和输入数据。
- 在Linux环境下进入api_tutorial\binary_input执行00-03脚本即可生成模型和输入数据
- 拷贝tmp_models和tmp_res到windows相同的目录下
### 使用命令行编译
- step_1: 执行命令，生成vs工程和对应的可执行文件
```shell
./build.bat [Debug|Release|RelWithDebInfo|MinSizeRel] #默认为Release
```
- step_2: 配置环境变量
```shell
$env:path+=";${NPU_SAMPLE_ROOT_DIR}\..\..\windows\simulator\bin"
```
- step_3: 执行sample（api_tutorial\binary_input）
```shell
.\api_tutorial\ortm_run_model\Release\sample_binary_input.exe -m api_tutorial/binary_input/tmp_models/resnet50_binary.ortm -i api_tutorial/binary_input/tmp_res/ILSVRC2012_val_00024327.bin
```
### 使用Visual Studio进行编译
-  step_1: 使用命令行生成vs工程，工程路径在samples\build_win\samples.sln
```shell
./build.bat 
```
-  step_2: 使用Visual Studio打开工程
-  step_3: 编译和调试(项目属性->调试->命令行参数 配置输入的模型和数据)
```shell
#配置的命令参数参考如下
-m api_tutorial/binary_input/tmp_models/resnet50_binary.ortm -i api_tutorial/binary_input/tmp_res/ILSVRC2012_val_00024327.bin
```
