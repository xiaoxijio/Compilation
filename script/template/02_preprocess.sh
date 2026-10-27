#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

# generate random input file in template, please implement as needed
cd ${CUR_DIR}
mkdir -p ./tmp_res/infer_data
python ${NPU_SAMPLE_ROOT_DIR}/script/gen_random_input.py --model ./tmp_models/{{MODEL_NAME}}.onnx \
    --output ./tmp_res/infer_data
