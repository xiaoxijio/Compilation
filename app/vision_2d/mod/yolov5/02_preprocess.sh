#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
mkdir -p tmp_res/infer_data/
cp ../../../../data/ImageNet/yuv420sp/yolo_src_640x640.yuv420sp tmp_res/infer_data/