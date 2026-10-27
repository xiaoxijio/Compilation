#!/bin/sh
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../../../..

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
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
}

function inference()
{
    cd ${CUR_DIR}

    if [ ${PLATFORM} == "simulator" ]; then
        set_run_sim_env
    else
        set_run_board_env
    fi
    if [ "$(uname -m)" = "aarch64" ]; then
        ./sample_mobilenetv2 -m ../../../../data/prebuild_ortm/mobilenetv2_rgbplanar.ortm
    else
        ./sample_mobilenetv2 -m ./tmp_models/mobilenetv2_rgbplanar.ortm
    fi
}

function main()
{
    inference
}

main $@
