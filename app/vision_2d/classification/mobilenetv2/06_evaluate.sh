#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

source ${NPU_SAMPLE_ROOT_DIR}/script/prepar_dataset.sh
prepar_ILSVRC2012_dataset

echo "check evaluate env"
check_file_dir_exist "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/ILSVRC2012/ILSVRC2012_img_val/"
ORTM_MODEL_NAME=$(ls ./tmp_models/*.ortm | grep -v 'fake_quant.ortm')

print_and_run_cmd \
python ${NPU_SAMPLE_ROOT_DIR}/script/evaluate_classification.py \
    --image_path ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/ILSVRC2012/ILSVRC2012_img_val/ \
    --resize 256 256 --crop 224 224 --target_shape 224 224 --convert_type YUV420SP  \
    --model_path ${ORTM_MODEL_NAME} --board_ip ${BOARD_IP} --port ${PORT}
