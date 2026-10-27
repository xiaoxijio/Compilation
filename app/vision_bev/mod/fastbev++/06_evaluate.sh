#!/bin/sh

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

source ${NPU_SAMPLE_ROOT_DIR}/script/prepar_dataset.sh
prepar_nuscenes_dataset fastbev++

# 数据预处理
cd "${NPU_SAMPLE_ROOT_DIR}/app/vision_bev/mod/common/evaluate"
python save_preprocessed_imgs.py --config fastbev --save_path ./preprocessed_imgs/
# 板端推理
cd ${CUR_DIR}
MODEL_PATH=${CUR_DIR}/tmp_models/fastbev++.ortm
python ${NPU_SAMPLE_ROOT_DIR}/script/evaluate_bev.py \
    --data_path ${CUR_DIR}/../common/evaluate/preprocessed_imgs \
    --model_path ${MODEL_PATH}  --board_ip ${BOARD_IP} --port ${PORT}
# mAP评估
cd "${NPU_SAMPLE_ROOT_DIR}/app/vision_bev/mod/common/evaluate"
python board_sample_inference.py --config configs/fastbev/fastbev-r50-cbgs.py --board_out_path ./preprocessed_imgs/