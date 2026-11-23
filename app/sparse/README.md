# S6：动态标定 ONNX → LQ560 ORTM

本目录只维护 S6 FastBEV 结构，不再使用 S2/LSS 的动态 scatter 改写和自定义 CPU 池化。
模型来源：`/home/fjm/tong/SparseDrive/ckpt/iter_3512.pth`，配置 `projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py`。它是训练中第3512步checkpoint，不是训练最终权重。

## 从 PTH 开始

在 SparseDrive 环境执行：

```bash
cd /home/fjm/tong/SparseDrive
conda activate sparsedrive
PYTHONPATH=. python tools/export_ep1_npu.py \
  --config projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py \
  --checkpoint ckpt/s6.pth --output onnx/s6/s6.onnx
```

默认即为上述配置、权重和输出路径；以后换权重请显式传 `--checkpoint`，换输出位置传 `--output`。默认做onnxsim简化，`--no-simplify`仅用于排查；`--check-only`复核现有ONNX。S6仅支持动态标定导出，不能套用固定LSS池化。

输出 `s6.onnx`、匹配的 `s6.calibration.npz`、`s6.validation.txt`。NPZ是验证/可视化所需的图像尺寸及标定来源说明，**不把其中的相机标定作为固定ONNX输入**。验证报告保留浮点误差和类别差异，不再生成临时JSON和warnings.json。

导出保留9个运行时输入，`image_wh`在图内固定为六路704×256：

| 输入 | FP32形状 | 内容 |
|---|---|---|
| img | 1×6×3×256×704 | 已完成项目归一化的六路图像 |
| projection_mat | 1×6×4×4 | 检测用 lidar→image，每帧真实标定/增强后的投影 |
| occ_ego2img | 1×6×4×4 | OCC用 ego→image，**不是旧的image→ego矩阵** |
| history_feature | 1×300×256 | 前一次推理返回的缓存 |
| history_anchor | 1×300×11 | 前一次返回的anchor |
| history_confidence | 1×300 | 前一次返回的置信度 |
| history_transform | 1×4×4 | 上一帧lidar坐标到当前lidar坐标的运动补偿 |
| time_interval | 1 | 秒 |
| history_valid | 1 | 首帧/切scene/无效间隔为0，正常有效历史为1 |

OCC训练接口依旧接收 `occ_img2ego`。FastBEV反向采样实际需要其逆矩阵；导出时改为直接输入正向投影，以省去NPU内的矩阵求逆。输入准备使用 `tools.ep1_common.occ_forward_projection` 的FP32计算，保持训练中 `affine_inverse` 的运算顺序。`tools/onnx_stage1.py` 已自动处理这个转换。板端也可直接提供真实的ego→image标定，必须包含与输入图像一致的内参和裁剪缩放等变换，不能直接把检测的lidar→image矩阵复制过去。

当前checkpoint实际只在 `z=-1` 单平面采样，预测头输出128×128×16、18类；不要把16个预测高度理解为16个采样平面。改成16层采样需要重新训练。

## 校准和编译

先生成连续帧输入。此处默认4帧，覆盖首帧、真实时间间隔、运动补偿、历史缓存及逐帧相机标定，只用于编译流程验证；它不足以证明全数据集INT8精度。

```bash
cd /home/fjm/tong/SparseDrive
PYTHONPATH=. python /home/fjm/tong/Compilation/app/sparse/prepare_calibration.py \
  --onnx onnx/s6/s6.onnx \
  --config projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py \
  --frames 4 --output /home/fjm/tong/Compilation/app/sparse/calibration

cd /home/fjm/tong/Compilation/app/sparse
bash 00_init.sh /home/fjm/tong/SparseDrive/onnx/s6/s6.onnx
bash 01_check.sh
bash 02_preprocess.sh
bash 03_omg.sh
```

目标由 `../../script/platform_config.sh` 决定，当前为 **LQ560ProV200**；SDK工具为2.1.6。生成板端ORTM不等于已经在实体板上完成运行测试。

| 文件 | 作用 |
|---|---|
| 00_init.sh | 暂存输入ONNX，根据真实输入形状重新生成input_cfg.yml |
| 01_check.sh | SDK检查原始导出图，并核对CSV，防止SDK返回0但存在不支持算子 |
| 02_preprocess.sh | 标准算子改写、四帧ORT数值比较、几何精度配置，再做SDK检查 |
| 03_omg.sh | 在自动清理的独立目录编译，成功后归档ORTM，并更新models下的快捷链接 |
| make_config.py | 生成SDK基础量化配置和9个输入定义 |
| prepare_calibration.py | 从真实连续帧生成编译输入，历史使用上一次ONNX实际输出 |
| prepare_lq560.py | 应用改写并验证，在本批目录保存model.onnx和几何FP16配置 |
| s6_graph.py | 02 编译适配所需的高维广播和 MatMul 降维、数值掩码、几何张量 FP16 选择 |
| 04_build.sh / 05_inference.sh / profile_report.py | 构建和运行 SDK PC 模拟器，按批次保存性能报告 |
| calibration/ | 重复编译需要的校准输入和dataset.txt，属于输入数据 |
| results/<模型名>/<运行目录>/model.onnx 与 models/s6.ortm | 最终编译用ONNX和板端模型；s6.calibration.npz保留配套验证元数据 |
| results/ | 最终检查、数值验证和编译日志 |

SDK产生的fake_quant模型、模型info JSON、GraphDump和编译工作目录在成功后清理；没有保留实验版模型或CPU算子库。

## 每批生成的 input_cfg.yml

- `quant.dataset.calib_path`：每行一帧的多输入二进制路径，列顺序必须严格匹配ONNX输入顺序；使用绝对路径，支持独立临时工作目录。
- `quant_batch_size=1`：模型按batch1导出，校准也用batch1，不是一次只允许一张相机图像。
- `activation: MaxMin / S8 / symmetric=false`：根据校准范围计算激活量化参数。S8是有符号8位，非对称量化允许非零zero point。
- `weight: MaxMin / S8 / per_channel=true`：卷积/线性权重逐通道量化。
- `quant_tensors`：02按实际图中名称生成，几何矩阵、投影坐标和必要几何常量使用FP16；图像主干等学习特征保持INT8。不是全模型FP16，也不是不加保护地把投影矩阵直接量化为INT8。
- `executor.enable_gpu=false / cpu_settings.thread_num=4`：PC校准和图执行使用CPU及4线程，不是板上NPU核心数设置。
- `enable_split_model_by_batch=false`：不按输入batch拆图；六相机维度不是batch切分开关。
- `inputs`：全部Binary/FP32，形状与ONNX一致；图像已有归一化，所以不再在此配置mean/std、Resize或RGB反转。矩阵和历史张量也不能套用图像预处理。

参考官方《LQ560V200 NPU工具使用指南》：量化张量精度配置（印刷页6）、GridSample（页39～40）、OMG检查/编译流程。实际解决了两个SDK限制：

1. `Round→Clip→Mul`被误识别为QAT，报qmax=43。改为等价Min/Max，正确处理Clip的可选边界。
2. FP16比较+bool转float虽然通过算子检查，后端无法打包。对仅用于浮点掩码的比较改为 `ceil(min(max(x-threshold,0),1))`；有限输入下等价于0/1比较结果，Where使用的真正布尔条件不改。

另外把相机投影MatMul二维化，将检测边界mask移到分组采样权重，保持特征求和顺序，减少重复乘法。所有改写均比较连续真实帧，不把“支持检查通过”等同于数值正确。

## 精度和性能的口径

浮点导出比较覆盖首帧、有效历史、重置和不同样本标定。第二样本有1个OCC体素类别翻转（共262144个），原始前两类logit分差约4.77e-7，最大logit误差约9.92e-5；记录该差异，并要求logits满足原3e-3容差且翻转位置的类别分差在局部误差范围内。不能声称所有类别逐位完全相同。

编译改写的四帧OCC logits误差为0，检测/缓存最大误差约1.42e-4。FP16/INT8量化后的全量mAP、NDS、mIoU尚需用代表性校准集及完整验证集单独评估，不能用这四帧或编译成功替代。

```bash
bash 04_build.sh
bash 05_inference.sh models/s6_dynamic.ortm
bash 05_inference.sh models/s6_fixed.ortm
```

## 本次实际结果

`iter_3512.pth`已完成实际编译，最终图1437个算子均为NPU。输出 `models/s6.ortm` 约99MiB，完整编译日志保留在results/compile.log。SDK确认两个外部FP32矩阵经过Preprocess后以FP16参与图内计算；输入表的FP16/FP32 Mismatch提示对应外部接口类型，并不代表内部矩阵被INT8量化。

官方单核性能模拟：约343.19ms，2.91383 FPS，未包含实体板的输入准备、传输及应用开销，**不是上板实测**。模拟统计没有CPU算子。主要耗时集中在transpose、reduction和向量运算，不能将当前结果称为全局最优性能。

SDK fake-quant首帧执行成功，7个输出均为有限值；与FP32的体素类别一致率约91.23%，检测框仍有明显差异。该一致率不是mIoU；缓存按原TopK顺序直接对比，排序变化会放大逐行最大误差。**当前仅完成编译及运行流程验证，INT8精度未验收**。需要代表性校准和全量验证，详细原始数值见results/quant_smoke.txt。


## 按模型保留编译和性能结果

四轮采样的大转置由 SparseDrive 导出器优化。LQ560 的编译图适配仍由 `02_preprocess.sh` 调用 `s6_graph.py` 完成。

`00_init.sh` 创建 `results/<S6_VARIANT>/<UTC时间_随机后缀>/`。后续 `01`–`03` 默认使用这个模型的 `latest` 链接，也可以通过 `S6_RUN_DIR` 指向指定批次。每批保留配置、算子检查表、数值比对、编译日志和 ORTM。成功编译后 `models/<S6_VARIANT>.ortm` 指向该批次的产物。原有顶层 results 日志保留原位。

`04_build.sh` 构建官方通用 PC 模拟器。每次运行 `05_inference.sh <ORTM>`，按 ORTM 的文件名创建独立性能目录，保存 `summary.txt`、原始 CSV 和执行日志。表内 FPS 和毫秒数是 SDK 单核模型估计。可把以前运行目录里的 ORTM 路径直接传给 05 来复测。

S7 的两段 ONNX、主机索引池化和编译入口见 SparseDrive 的 `docs/s7_onnx_compile.md`。
