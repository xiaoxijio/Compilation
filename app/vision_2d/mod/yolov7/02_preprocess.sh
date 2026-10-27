#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
single_img=$(head -n 1 ./tmp_res/quant_dataset/coco2017_50.txt)
python ${NPU_SAMPLE_ROOT_DIR}/script/preprocess.py --image_path ${single_img} --save_path ./tmp_res/infer_data/ \
    --letter_box --target_shape 640 640 --convert_type YUV420SP
