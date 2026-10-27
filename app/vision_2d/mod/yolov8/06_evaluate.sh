#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

source ${NPU_SAMPLE_ROOT_DIR}/script/prepar_dataset.sh
prepar_COCO2017_dataset

echo "check evaluate env"
check_file_dir_exist "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/COCO2017/val2017/"
ORTM_MODEL_NAME=$(ls ./tmp_models/*.ortm | grep -v 'fake_quant.ortm')
# 板端推理+后处理+结果保存为txt文件
rm -rf tmp_res/pred/ tmp_res/result.json
print_and_run_cmd \
python ${NPU_SAMPLE_ROOT_DIR}/script/post_process_yolo.py \
    --image_path ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/COCO2017/val2017/ \
    --letter_box --target_shape 640 640 --convert_type YUV420SP  \
    --model_path ${ORTM_MODEL_NAME} --board_ip ${BOARD_IP} --port ${PORT} \
    --postprocess_type yolov8 \
    --pred_dir ${CUR_DIR}/tmp_res/pred/

python ${NPU_SAMPLE_ROOT_DIR}/script/yolo2coco_pred.py
python ${NPU_SAMPLE_ROOT_DIR}/script/get_coco_metrice.py
