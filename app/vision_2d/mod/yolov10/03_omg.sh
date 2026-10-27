
#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

MODEL_PATH="./tmp_models/yolov10s.onnx"
echo "Check quant env"
check_file_dir_exist "${MODEL_PATH}"
check_file_dir_exist "${CUR_DIR}/tmp_res/quant_dataset/images/"

OUTPUT_MODEL=${MODEL_PATH:0:-5}.ortm

if [[ $(check_target "${TARGET}") = "0" ]]; then
    config="input_cfg_0.yml"
else
    config="input_cfg_1.yml"
fi

print_and_run_cmd \
${OMG}                                              \
    --model             ${MODEL_PATH}               \
    --config            ${config}                   \
    --output_model      ${OUTPUT_MODEL}             \
    --target            ${TARGET}                   \
    --log_level         info