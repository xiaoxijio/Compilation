
#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH="./tmp_models/model.safetensors"
echo "Check Model"
check_file_dir_exist "${MODEL_PATH}"

OUTPUT_MODEL=./tmp_models/tinyllama

${OMG}                                           \
    --safetensors       ${MODEL_PATH}            \
    --config            ./tmp_res/input_cfg.yml  \
    --hf-config         ./tmp_res/config.json    \
    --output_model      ${OUTPUT_MODEL}          \
    --target            ${TARGET}                \
    --log_level         info