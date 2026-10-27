#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

cd ${CUR_DIR}
mkdir -p tmp_models 
mkdir -p tmp_res
rm -rf tmp_models/* tmp_res/*

cp ${NPU_SAMPLE_ROOT_DIR}/data/model/yolov7.onnx ./tmp_models -rf
if [ ! -d "./tmp_res/quant_dataset" ]; then
    tar -xzf ${NPU_SAMPLE_ROOT_DIR}/data/quant_dataset.tar.gz -C ./tmp_res/
fi
sed -i "s#^\./#${CUR_DIR}/tmp_res/quant_dataset/images/#" ./tmp_res/quant_dataset/imageNet_50.txt
sed -i "s#^\./#${CUR_DIR}/tmp_res/quant_dataset/images/#" ./tmp_res/quant_dataset/coco2017_50.txt
echo "init over"
