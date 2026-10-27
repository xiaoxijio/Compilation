#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../..

source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

function set_run_sim_env()
{
    SIM_LIB_PATH=
    if [ -d "/avp/npu/simulator" ]; then
        SIM_LIB_PATH=/avp/npu/simulator/lib
    elif [ -d "${NPU_SAMPLE_ROOT_DIR}/../../linux/simulator" ]; then
        SIM_LIB_PATH=${NPU_SAMPLE_ROOT_DIR}/../../linux/simulator/lib
    elif [ -e ${NPU_SAMPLE_ROOT_DIR}/../output/lib/libotsimulator.so ]; then
        SIM_LIB_PATH=${NPU_SAMPLE_ROOT_DIR}/../../linux/simulator/lib
    else
        SIM_LIB_PATH=${NPU_TOOLCHAIN_PATH}/linux/simulator/lib
    fi
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SIM_LIB_PATH}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
}

function set_run_board_env()
{
    OPENCV_LIB_PATH="${NPU_SAMPLE_ROOT_DIR}/third_party/opencv/lib"
    case "$LD_LIBRARY_PATH" in
        "*opencv*")   ;;
        *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENCV_LIB_PATH}" ;;
    esac
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
}

function inference()
{
    cd ${CUR_DIR}
    HEAD_PARAMETERS_PATH=${CUR_DIR}/../common/head_parameters
    DATA_PATH=${NPU_SAMPLE_ROOT_DIR}/data/got10k/
    OUT_PATH=${NPU_SAMPLE_ROOT_DIR}/data/got10k/output

    if [ ${PLATFORM} == "simulator" ]; then
        ORTM_MODEL_NAME=${CUR_DIR}/tmp_models/mixformer_backbone.ortm
        set_run_sim_env
    else
        ORTM_MODEL_NAME=../../../data/prebuild_ortm/mixformer_backbone.ortm    
        set_run_board_env
    fi
    
    # for this sample FRAME_CNT can be 1-60
    FRAME_CNT=3
    ./sample_mixformer                \
        -m ${ORTM_MODEL_NAME}         \
        -v ${DATA_PATH}               \
        -p ${HEAD_PARAMETERS_PATH}    \
        -s ${OUT_PATH}                \
        -n ${FRAME_CNT}
    echo -e "\033[32mExecute finished and visualization complete. Results are in ${OUT_PATH}\033[0m"   
}

function main()
{
    inference
}

main $@
