#!/bin/sh

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

DATA_SAVE_PATH=${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset
RELATIVE_PATH="lightstereo/sceneflow_datasets.tgz"
DATA_PATH=${DATA_SAVE_PATH}/sceneflow_datasets.tgz

if [ ! -f ${DATA_PATH} ]; then
    fetch_evaluation_datasets "${RELATIVE_PATH}" "${DATA_SAVE_PATH}"
else
    echo "sceneflow_datasets.tgz file exist!"
fi

echo "Unzipping sceneflow_datasets.tgz"
tar -xzf ${DATA_PATH} -C ${NPU_SAMPLE_ROOT_DIR}/app/stereo/lightstereo/evaluate
if [ $? -eq 0 ]; then
    echo "Unzipping complete"
else
    echo "Unzipping fail"
    exit 1
fi

MODEL_PATH=${CUR_DIR}/tmp_models/lightstereo.ortm

python ${NPU_SAMPLE_ROOT_DIR}/script/evaluate_stereo.py \
    --data_path ${CUR_DIR}/evaluate/sceneflow_datasets \
    --model_path ${MODEL_PATH}  --board_ip ${BOARD_IP} --port ${PORT}
