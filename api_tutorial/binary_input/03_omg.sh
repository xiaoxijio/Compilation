
#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

OUTPUT_MODEL=./tmp_models/resnet50_binary.ortm

${OMG}                                              \
    --model             ./tmp_models/resnet50_binary.onnx \
    --config            input_cfg.yml            \
    --output_model      ${OUTPUT_MODEL}             \
    --target            ${TARGET}                   \
    --log_level         info