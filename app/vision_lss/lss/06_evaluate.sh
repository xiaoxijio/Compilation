#!/bin/sh

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}

# 拉取评估数据集
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/prepar_dataset.sh
prepar_nuscenes_dataset lss
# 数据预处理
DATA_ROOT=${CUR_DIR}/data/nuscenes
SAVE_PATH=${CUR_DIR}/data/inputs
python3 ../common/data_init/parse_data.py --pub ${ENABLE_OPENSOURCE_ONNX} --dataroot ${DATA_ROOT} --savepath ${SAVE_PATH}

MODEL_PATH=${CUR_DIR}/tmp_models/lss.ortm
python ${NPU_SAMPLE_ROOT_DIR}/script/evaluate_lss.py                \
    --data_input ${CUR_DIR}/data/inputs                             \
    --data_output ${CUR_DIR}/data/outputs                           \
    --model_path ${MODEL_PATH}                                      \
    --board_ip ${BOARD_IP} --port ${PORT}                           \
    --pub ${ENABLE_OPENSOURCE_ONNX}

# 可视化（需用到评估的推理结果）
# 生成图片
python ../common/view/vis_model_infer_results.py --pub $ENABLE_OPENSOURCE_ONNX
# 图片生成视频
python ../common/view/images2video.py

python ../common/evaluate/eval.py --pub ${ENABLE_OPENSOURCE_ONNX} --data_root=data/nuscenes --board_out=data/outputs/