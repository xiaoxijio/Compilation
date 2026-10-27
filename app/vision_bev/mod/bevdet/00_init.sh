#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}

# 选择开源bevdet/魔改bevdet ONNX
# ENABLE_OPENSOURCE_ONNX=0 # 选择开源bevdet/魔改bevdet ONNX, 1:开源，0: 魔改
# 为方便统一设置，将该配置项迁移至 ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

BEVDET_ONNX_NAME=bevdet-efficientnetHz-cbgs-relu6.onnx
if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
BEVDET_ONNX_NAME=bevdet-r50-cbgs.onnx
fi

mkdir -p tmp_models 
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/model/bevdet/${BEVDET_ONNX_NAME}  ${CUR_DIR}/tmp_models/bevdet.onnx

cd ${NPU_SAMPLE_ROOT_DIR}/data
NUSCENSES_DATASET=nuscenes
if [ ! -d ${NUSCENSES_DATASET} ] ; then
    tar -zxvf ${NUSCENSES_DATASET}.tar.gz
fi
cd ${CUR_DIR}