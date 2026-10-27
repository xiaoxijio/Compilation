
#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH=${CUR_DIR}/tmp_models/fastbev++.onnx
OUTPUT_MODEL=${MODEL_PATH:0:-5}.ortm

if [[ $(check_target "${TARGET}") = "0" ]]; then
    echo "${TARGET} not support, please set ${TARGET_LIST2[@]}"
    exit
fi

${OMG}                                              \
    --model             ${MODEL_PATH}              \
    --config            input_cfg.yml               \
    --output_model      ${OUTPUT_MODEL}             \
    --target            ${TARGET}                   \
    --log_level         info