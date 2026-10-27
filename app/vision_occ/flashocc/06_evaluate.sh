#!/bin/sh

CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

cd ${CUR_DIR}
source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh
# 准备数据集
source ${NPU_SAMPLE_ROOT_DIR}/script/prepar_dataset.sh
prepar_nuscenes_dataset flashocc $ENABLE_OPENSOURCE_ONNX

# 数据预处理
echo "data preprocess start"
cd "${NPU_SAMPLE_ROOT_DIR}/app/vision_occ/evaluate"
gts_opt=${NPU_SAMPLE_ROOT_DIR}/app/vision_occ/evaluate/data/nuscenes/gts_opt
if [ ! -d ${gts_opt} ] ; then
    python generate_gts_128x128.py
fi
if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
    python save_preprocessed_imgs.py --config FlashOCC_Board/configs/flashocc/flashocc-r50-M0.py  --save_path data/inputs
else
    python save_preprocessed_imgs.py --config FlashOCC_Board/configs/flashocc/flashocc-r50-M0-opt.py  --save_path data/inputs
fi
echo "data preprocess end"
# 数据推理
echo "data inference start"
cd ${CUR_DIR}
MODEL_PATH=${CUR_DIR}/tmp_models/flashocc.ortm
python ${NPU_SAMPLE_ROOT_DIR}/script/evaluate_flashocc.py                \
    --data_input ${CUR_DIR}/../evaluate/data/inputs                      \
    --data_output ${CUR_DIR}/../evaluate/data/outputs                    \
    --model_path ${MODEL_PATH}  --board_ip ${BOARD_IP} --port ${PORT}
echo "data inference end"

# 评估
echo "model eva start"
cd ${CUR_DIR}/../evaluate
if [ ${ENABLE_OPENSOURCE_ONNX} -eq 1 ]; then
    python FlashOCC_Board/test_flashocc.py FlashOCC_Board/configs/flashocc/flashocc-r50-M0.py --eval mAP --board-outs data/outputs/
else
    python FlashOCC_Board/test_flashocc.py FlashOCC_Board/configs/flashocc/flashocc-r50-M0-opt.py --eval mAP --opt --board-outs data/outputs/
fi
echo "model eva end"
cd ${CUR_DIR}