#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

cd ${CUR_DIR}
mkdir -p tmp_models 
mkdir -p tmp_res
cp ${NPU_SAMPLE_ROOT_DIR}/data/model/abs.onnx ./tmp_models -rf
cp ${NPU_SAMPLE_ROOT_DIR}/data/custom_abs/* ./tmp_res -rf
sed -i 's/^/.\/tmp_res\//' ./tmp_res/*.txt