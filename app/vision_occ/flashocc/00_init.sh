#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}

# 选择开源 flashocc/魔改 flashocc ONNX
# ENABLE_OPENSOURCE_ONNX=0 # 选择开源 flashocc/魔改 flashocc ONNX, 1:开源，0: 魔改
# 为方便统一设置，将该配置项迁移至 ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

FLASHOCC_ONNX_NAME=flashocc_opt.onnx
if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
FLASHOCC_ONNX_NAME=flashocc-r50-m0_simplified.onnx
fi

mkdir -p tmp_models
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/model/flashocc/${FLASHOCC_ONNX_NAME} ${CUR_DIR}/tmp_models/flashocc.onnx

cd ${NPU_SAMPLE_ROOT_DIR}/data
NUSCENSES_DATASET=flashocc
if [ ! -d ${NUSCENSES_DATASET} ] ; then
    tar -zxvf ${NUSCENSES_DATASET}.tar.gz
fi
NUSCENSES_DATASET=nuscenes
if [ ! -d ${NUSCENSES_DATASET} ] ; then
    tar -zxvf ${NUSCENSES_DATASET}.tar.gz
fi
cd ${CUR_DIR}