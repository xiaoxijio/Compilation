#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

mkdir -p tmp_models
mkdir -p tmp_res

echo "Check Data Init"
source ${CUR_DIR}/../common/common.sh
prepar_llm_model qwen
MODEL_SAFETENSORS=${NPU_SAMPLE_ROOT_DIR}/data/llm/qwen/model.safetensors
check_file_dir_exist "${MODEL_SAFETENSORS}"

if [ ! -f ./tmp_models/model.safetensors ]; then
    ln -s ${MODEL_SAFETENSORS} ./tmp_models/model.safetensors
fi
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/llm/vocab.json               ${CUR_DIR}/tmp_res/
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/llm/qwen/input_cfg.yml       ${CUR_DIR}/tmp_res/
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/llm/qwen/config.json         ${CUR_DIR}/tmp_res/
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/llm/qwen/tokenizer.model     ${CUR_DIR}/tmp_res/
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/llm/qwen/DeepSeek-R1-Distill-Qwen-1.5B-fp16.bin    ${CUR_DIR}/tmp_res/