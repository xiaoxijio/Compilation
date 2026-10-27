
#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH="./tmp_models/{{MODEL_NAME}}.onnx"
echo "Check quant env"
check_file_dir_exist "${MODEL_PATH}"
check_file_dir_exist "${CUR_DIR}/tmp_res/quant_dataset/images/"

OUTPUT_MODEL="./tmp_models/{{MODEL_NAME}}.ortm"
config="input_cfg.yml"

print_and_run_cmd \
${OMG}                                              \
    --model             ${MODEL_PATH}               \
    --config            ${config}                   \
    --output_model      ${OUTPUT_MODEL}             \
    --target            ${TARGET}                   \
    --log_level         info