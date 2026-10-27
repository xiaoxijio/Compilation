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
    OPENCV_LIB_PATH="${NPU_SAMPLE_ROOT_DIR}/third_party/opencv/lib/aarch64"
    case "$LD_LIBRARY_PATH" in
        "*opencv*")   ;;
        *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENCV_LIB_PATH}" ;;
    esac
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
}

function inference()
{
    cd ${CUR_DIR}
    
    if [ ${PLATFORM} == "simulator" ]; then
        ORTM_MODEL_NAME=./tmp_models/flashocc.ortm
        DATA_PATH=${NPU_SAMPLE_ROOT_DIR}/data/bev/
        set_run_sim_env
    else
        ORTM_MODEL_NAME=${NPU_SAMPLE_ROOT_DIR}/data/prebuild_ortm/flashocc.ortm
        DATA_PATH=${NPU_SAMPLE_ROOT_DIR}/data/bev/
        set_run_board_env
    fi

    ./sample_flashocc         \
        -m ${ORTM_MODEL_NAME} \
        -i ${DATA_PATH}       \
        -n 1

}

function main()
{
    inference
}

main $@
