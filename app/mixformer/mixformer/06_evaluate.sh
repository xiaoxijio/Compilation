#!/bin/sh

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}
# 拉取评估数据集
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/prepar_dataset.sh
prepar_GOT10K_dataset

MODEL_PATH=${CUR_DIR}/tmp_models/mixformer_backbone.ortm

python ${NPU_SAMPLE_ROOT_DIR}/script/evaluate_mixformer.py                              \
    --board_ip ${BOARD_IP}                                                              \
    --port ${PORT}                                                                      \
    --model_path ${MODEL_PATH}                                                          \
    --video_path ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/GOT_10K                   \
    --save_path ${CUR_DIR}/tmp_res/tracking_results                                            \
    --para_path ${CUR_DIR}/../common/head_parameters

cd ${CUR_DIR}/../common/evaluate

python tracking/analysis_results.py                 \
    --tracker_name mixformer2_vit_online_onnx       \
    --dataset_name got10k_val                       \
    --tracker_param 224_depth4_mlp1_score