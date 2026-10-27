#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}

# ENABLE_OPENSOURCE_ONNX=1 # 选择开源lightstereo/魔改lightstereo ONNX, 1:开源，0: 魔改
# 为方便统一设置，将该配置项迁移至 ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

LIGHTSTEREO_ONNX_NAME=lightstereo_s+_sceneflow_opt_qdq_epoch209_int8_sim.onnx
LIGHTSTEREO_YAML_NAME=input_cfg.yml
if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
LIGHTSTEREO_ONNX_NAME=lightstereo.onnx
LIGHTSTEREO_YAML_NAME=input_cfg_1.yml
fi
mkdir -p tmp_models
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/model/lightstereo/${LIGHTSTEREO_ONNX_NAME}  ${CUR_DIR}/tmp_models/lightstereo.onnx
cp ./${LIGHTSTEREO_YAML_NAME} ./tmp_models/input_cfg.yml
cd ${NPU_SAMPLE_ROOT_DIR}/data
NUSCENSES_DATASET=sceneflow_cali8
if [ ! -d ${NUSCENSES_DATASET} ] ; then
    tar -zxvf ${NUSCENSES_DATASET}.tar.gz
fi
cd ${CUR_DIR}