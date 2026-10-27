#!/bin/sh
echo "OMG check model"
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

if [[ $(check_target "${TARGET}") = "0" ]]; then
    echo "${TARGET} not support, please set ${TARGET_LIST2[@]}"
    exit
fi

${OMG}                                              \
    --model         ./tmp_models/lightstereo.onnx   \
    --config        ./tmp_models/input_cfg.yml      \
    --target        ${TARGET}                       \
    --check