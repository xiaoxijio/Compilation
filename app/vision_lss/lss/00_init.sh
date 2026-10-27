#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

# ENABLE_OPENSOURCE_ONNX=0 # 选择开源LSS/魔改LSS ONNX, 1:开源，0: 魔改
# 为方便统一设置，将该配置项迁移至 ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

LSS_ONNX_NAME=lss_sim_del_reshape_opt.onnx
CONFIG_FILE=input_cfg_opt.yml

if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
    LSS_ONNX_NAME=lss_sim_del_reshape_v1.onnx
    CONFIG_FILE=input_cfg_pub.yml
fi

cd ${CUR_DIR}
cp $CONFIG_FILE input_cfg.yml
mkdir -p tmp_models
cp -rf ${NPU_SAMPLE_ROOT_DIR}/data/model/lss/${LSS_ONNX_NAME}    ${CUR_DIR}/tmp_models/lss.onnx

rm -rf tmp_res/
mkdir -p tmp_res/quant_dataset/
tar -zxvf ${NPU_SAMPLE_ROOT_DIR}/data/bev_quant_dataset.tar.gz -C tmp_res/
# 生成量化数据集
DATA_ROOT=${CUR_DIR}/tmp_res/bev_quant_dataset
SAVE_PATH=${CUR_DIR}/tmp_res/quant_dataset

python3 ../common/data_init/get_calibration_data.py --pub ${ENABLE_OPENSOURCE_ONNX} --quant_data ${DATA_ROOT} --out_path ${SAVE_PATH}

cd ${NPU_SAMPLE_ROOT_DIR}/data
NUSCENSES_DATASET=nuscenes
if [ ! -d ${NUSCENSES_DATASET} ] ; then
    tar -zxvf ${NUSCENSES_DATASET}.tar.gz
fi
cd ${CUR_DIR}
echo init lss sample success

