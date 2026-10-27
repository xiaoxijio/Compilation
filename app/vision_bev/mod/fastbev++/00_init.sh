#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}

# ENABLE_OPENSOURCE_ONNX=0 # 选择开源fastbev++/魔改fastbev++ ONNX, 1:开源，0: 魔改
# 为方便统一设置，将该配置项迁移至 ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

FASTBEV_ONNX_NAME=quant_fastbev-eb0-cbgs_op14_0311_mAP2670_fix_sim.onnx
if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
FASTBEV_ONNX_NAME=fastbev-r50-cbgs_op14_mAP2905_fix_sim.onnx
fi
mkdir -p tmp_models
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/model/fastbev++/${FASTBEV_ONNX_NAME}  ${CUR_DIR}/tmp_models/fastbev++.onnx

cd ${NPU_SAMPLE_ROOT_DIR}/data
NUSCENSES_DATASET=nuscenes
if [ ! -d ${NUSCENSES_DATASET} ] ; then
    tar -zxvf ${NUSCENSES_DATASET}.tar.gz
fi
cd ${CUR_DIR}