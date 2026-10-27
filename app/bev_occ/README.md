# BEV OCC：模型编译与 PC 仿真

进入示例目录并激活 SDK Python 环境：

```bash
cd /home/tong/LQ560_SDK_V200R001C00SPC060/samples/app/bev_occ
source /home/tong/anaconda3/bin/activate sdk
```

## 动态映射模型 g3

```bash
bash 00_init.sh g3
python prepare_mapping_calibration.py --onnx g3
bash 01_check.sh g3
bash 03_omg.sh g3
bash 04_build.sh
bash 05_inference.sh --onnx g3
```

先将 `g3.onnx` 和 `g3.inputs.npz` 放到 `samples/data/model/bev_occ/`。
`00` 将 ONNX 和存在的同名/同前缀配套 NPZ 复制到 `tmp_models/`。
准备脚本自动选择 `tmp_models/g3.onnx`、`tmp_models/g3.inputs.npz`，
生成 `calibration/g3/` 和 `input_cfg/g3.yml`。
映射输入请求至少 FP16 精度，避免 S8 量化改变索引；SDK 部分索引路径可能保留 FP32。

NPZ 优先匹配 `<模型名>.inputs.npz`，其次 `<模型名>.npz`，再查找带 `. / _ / -`
分隔符的同前缀 NPZ；候选不唯一时要求显式 `--npz`，不会猜测。
可重复传入 `--npz` 使用多个实际样本。
重复执行 `python prepare_mapping_calibration.py --onnx g3` 会覆盖 `calibration/g3/`
和 `input_cfg/g3.yml`，旧目录内的残留样本会一并清理。输入校验通过后才替换旧数据。
校准数据无需更新时，也可以跳过准备脚本，直接执行 `01/03`。
需要保留原实验并重新生成校准数据时，显式指定新的输出名称：

```bash
python prepare_mapping_calibration.py --onnx g3 \
  --out-dir calibration/g3_new --config input_cfg/g3_new.yml
bash 01_check.sh g3 --config input_cfg/g3_new.yml
bash 03_omg.sh g3 --config input_cfg/g3_new.yml
bash 05_inference.sh --onnx g3 --tag g3_new
```

映射准备脚本默认覆盖指定的输出目录和配置。`03` 会更新同名 ORTM，并保存实际使用的配置供 `05` 自动归档。

## 固定几何模型 g3_nomap

先将 `g3_nomap.onnx` 放到 `samples/data/model/bev_occ/`，不要求配套 NPZ。

```bash
bash 00_init.sh g3_nomap
bash 01_check.sh g3_nomap
bash 03_omg.sh g3_nomap
bash 04_build.sh
bash 05_inference.sh --onnx g3_nomap
```

固定几何模型无需导出 NPZ，无需手动执行 `prepare_calibration.py`。
`00` 自动生成 `input_cfg/g3_nomap.yml`（已有配置则保留），使用
`calibration/dataset_bevdet.txt` 图像 binary 清单；清单不存在时从 SDK 自带的
`samples/data/ImageNet/dataset_bevdet.txt` 自动生成，与原固定几何流程一致。
图像已完成预处理，脚本检查尺寸与模型输入匹配。

INT8 编译仍需要图像来统计激活范围；这些是量化校准样本，与相机内外参标定无关。
`prepare_calibration.py` 是内部共用工具：单图像模型不带 `--npz` 时使用图像清单，
显式提供 NPZ 时也支持提取输入。日常运行固定几何模型无需手动调用它。
图像校准集应与实际模型预处理和场景匹配，现有示例只用于跑通流程，不代表量化精度通过。

默认量化参数（MaxMin、S8、权重 per-channel）内置在代码中，
不需要事先创建 `input_cfg/input_cfg.yml`。需要自定义基础参数时，
可显式传入 `--base-config input_cfg/自定义模板.yml`；指定文件不存在时会报告明确错误。

## 自动定位与目录

以 `--onnx g3` 为例，默认文件对应关系如下：

| 用途 | 路径 |
| --- | --- |
| ONNX | `tmp_models/g3.onnx` |
| 配套 NPZ | `tmp_models/g3.inputs.npz`（优先） |
| 映射校准数据 | `calibration/g3/` |
| 量化配置 | `input_cfg/g3.yml` |
| 正式模型 | `tmp_models/g3.ortm` |
| 实际编译配置快照 | `tmp_models/g3.compile.yml` |
| 仿真程序 | `build/ortm_run_sim_model` |
| 仿真结果 | `results/g3/<run-id>/` |

`--onnx` 支持模型名、带 `.onnx` 的文件名和完整路径。
`05 --onnx g3` 自动选择同名 ORTM、配置和结果标签 `g3`；
`--tag`、`--model`、`--model-name`、`--compile-config` 等显式选项仍保留。
`03` 成功后将实际编译配置保存为 `tmp_models/g3.compile.yml`，仿真归档优先使用它；
没有编译快照时只归档当前同名配置，并在 metadata 标注尚未确认它与旧 ORTM 一致。
更换校准配置后，应重新执行 `03`，已有 ORTM 不会自动重新编译。

- `tmp_models/`：ONNX、配套 NPZ、正式 ORTM、编译配置快照及编译信息。
- `input_cfg/`：各模型自动生成的 `<模型名>.yml`。
- `calibration/`：映射 binary、dataset 清单及 manifest，固定模型共用的图像清单。
- `build/`：PC 仿真程序，`04` 构建，`05` 使用。
- `results/<模型名>/<run-id>/`：日志、性能摘要和来源信息。

`01/03` 默认读取 `input_cfg/<模型名>.yml`，可用 `--config` 覆盖。
`04` 与模型无关，首次使用或删除 build 后执行即可。
`02_preprocess.sh` 是占位步骤，上述流程无需执行。
TARGET 沿用 `../../script/platform_config.sh`，当前为 `LQ560PlusV200`。

两种模型都完成性能仿真后，可生成对比报告：

```bash
bash 06_compare.sh --baseline g3_nomap
```

报告写入 `results/comparison.txt` 和 `results/comparison.csv`。

## 验证范围

`05` 默认性能估算，不读取图像，不代表板端实测或数值精度通过。
固定几何的 infer/dump 模式需额外传入 `--input`，仅支持一个 FP32 `[6,3,256,704]` binary；
22 输入 mapping 模型尚未适配真实数值推理。
不包含检测后处理、OCC 可视化或 nuScenes 指标计算。
`test_sim.py` 包含运行状态、输出完整性、性能报告和自动路径选择的回归测试。
`test_calibration.py` 检查无模板时生成映射配置、固定模型图像清单及缺少自定义模板的报错。

```bash
python -m unittest test_calibration.py test_sim.py
```
