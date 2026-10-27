#!/bin/bash
echo "OMG check model"
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH="./tmp_models/yolov5s.onnx"
check_file_dir_exist "${MODEL_PATH}"
check_file_dir_exist "${CUR_DIR}/tmp_res/quant_dataset/images/"

if [[ $(check_target "${TARGET}") = "0" ]]; then
    config="input_cfg_0.yml"
else
    config="input_cfg_1.yml"
fi

print_and_run_cmd \
${OMG}                                              \
    --model             ${MODEL_PATH}               \
    --config            ${config}                   \
    --target            ${TARGET}                   \
    --check