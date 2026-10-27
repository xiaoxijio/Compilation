
#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

OUTPUT_MODEL=./tmp_models/custom_abs.ortm

${OMG}                                              \
    --model             ./tmp_models/abs.onnx        \
    --config            input_cfg.yml            \
    --output_model      ${OUTPUT_MODEL}             \
    --target            ${TARGET}                   \
    --log_level         info