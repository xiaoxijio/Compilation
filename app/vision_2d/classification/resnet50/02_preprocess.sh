#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
single_img=$(head -n 1 ./tmp_res/quant_dataset/imageNet_50.txt)
python ${NPU_SAMPLE_ROOT_DIR}/script/preprocess.py --image_path ${single_img} --save_path ./tmp_res/infer_data/ \
    --resize 256 256 --crop 224 224 --target_shape 224 224 --convert_type YUV420SP
