#!/bin/bash
echo "OMG check model"
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH="./tmp_models/resnet50_binary.onnx"
check_file_dir_exist "${MODEL_PATH}"
check_file_dir_exist "${CUR_DIR}/tmp_res/quant_dataset/images/"

${OMG}                                              \
    --model             ${MODEL_PATH}               \
    --config            input_cfg.yml               \
    --target            ${TARGET}                   \
    --check