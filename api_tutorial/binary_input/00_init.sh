#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

cd ${CUR_DIR}
mkdir -p tmp_models 
mkdir -p tmp_res 
cp ${NPU_SAMPLE_ROOT_DIR}/data/model/resnet50_binary.onnx ./tmp_models -rf
cp ${NPU_SAMPLE_ROOT_DIR}/data/ImageNet/binary/ILSVRC2012_val_00024327.bin ./tmp_res -rf
cp ${NPU_SAMPLE_ROOT_DIR}/data/ImageNet/dataset_5.txt ./tmp_res -rf
cp ${NPU_SAMPLE_ROOT_DIR}/data/ImageNet/dataset_1.txt ./tmp_res -rf
cp ${NPU_SAMPLE_ROOT_DIR}/data/ImageNet/pic ./tmp_res -rf
sed -i 's/^\./.\/tmp_res/' ./tmp_res/dataset_*.txt