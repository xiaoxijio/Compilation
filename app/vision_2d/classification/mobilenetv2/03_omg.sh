
#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH=$(ls ./tmp_models/*.onnx)
OUTPUT_MODEL=${MODEL_PATH:0:-5}.ortm
echo "Check quant env"
check_file_dir_exist "${MODEL_PATH}"
check_file_dir_exist "${CUR_DIR}/tmp_res/quant_dataset/images/"

${OMG}                                              \
    --model             ${MODEL_PATH}               \
    --config            input_cfg.yml               \
    --output_model      ${OUTPUT_MODEL}             \
    --target            ${TARGET}                   \
    --log_level         info