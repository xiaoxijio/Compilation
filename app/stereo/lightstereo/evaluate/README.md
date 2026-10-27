## 一、环境配置
与 yolo sample一致

## 二、评估
### 1. 环境准备
在samples/script/platform_config.sh中对板端IP以及Port进行配置(此处示例IP:10.16.81.27,Port:1234)
进入到IP为10.16.81.27板端环境，在mount路径下执行命令：
``` shell
./avp_server -p 1234 (1234为监听端口号)
```
板端出现如下打印即成功建立连接
``` shell
Listen on port 1234
``` 
### 2. 运行测试集，获得模型输出
在stereo目录下执行
```shell
    ./06_evaluate.sh
```
执行上述脚本后，会在 samples/app/stereo/lightstereo/evaluate 目录下生成测试集的模型输出结果
### 3. 板端输出精度验证
```shell
# lightstereo
python samples/app/stereo/lightstereo/evaluate/evaluate.py --data_path 数据集路径
```
