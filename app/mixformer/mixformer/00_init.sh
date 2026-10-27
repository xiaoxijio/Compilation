#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}


MIXFORMER_ONNX_NAME=mixformer_v2_backbone.onnx
mkdir -p tmp_models
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/model/${MIXFORMER_ONNX_NAME}  ${CUR_DIR}/tmp_models/mixformer_v2_backbone.onnx

QUANT_DATA_PATH=${NPU_SAMPLE_ROOT_DIR}/data/mixformer_quant_dataset.tar.gz
DEMO_DATA_PATH=${NPU_SAMPLE_ROOT_DIR}/data/got10k.tar.gz
if [ ! -f "${QUANT_DATA_PATH}" ]; then
    echo "quant data not exist!"
fi
mkdir -p tmp_res
tar -zxf ${QUANT_DATA_PATH} -C  ${CUR_DIR}/tmp_res
tar -zxf ${DEMO_DATA_PATH} -C  ${NPU_SAMPLE_ROOT_DIR}/data

cd ${CUR_DIR}