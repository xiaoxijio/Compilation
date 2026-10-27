#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0); pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

source ${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh

BUILD_TYPE=Release

JOB_NUM=8

function build_clean()
{
    cd ${CUR_DIR}
    rm -rf build
    rm -rf output
    rm -rf GraphDump
    rm -rf LayerDump
    rm -rf QuantDump
    rm -rf ortm_run*model
}

function show_help()
{
    echo ""
    echo "./build.sh [-c]"
    echo "options:"
    echo "        -c : clean up. "
    echo ""
}

function parse_args()
{
    ARGS=$(getopt -o ch --long clean -n"$0" -- "$@")
    eval set -- "$ARGS"
    while [ -n "$1" ]; do
        case $1 in
        -h | \?)    show_help ; exit 0 ;;
        -c | clean) build_clean ; echo "clean up" ; exit 0 ;;
        --) ;;
        *)
            echo "'$1' is an unkonwn option" ; show_help ; shift ;;
        esac
        shift
    done
}


function build_exe()
{
    if [ ${PLATFORM} == "board" ]; then
        CROSS_COMPILE_FILE="${NPU_SAMPLE_ROOT_DIR}/cmake/cross_compile_aarch64_linux.cmake"
        SAMPLE_BUILD_TARGET="board"
    elif [ ${PLATFORM} == "simulator" ]; then
        SAMPLE_BUILD_TARGET="sim"
    fi

    cd ${CUR_DIR}

    cmake -B build                                      \
        -DCMAKE_TOOLCHAIN_FILE=${CROSS_COMPILE_FILE}    \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                \
        -DSAMPLE_BUILD_TARGET=${SAMPLE_BUILD_TARGET}
    cmake --build build -j${JOB_NUM}
    cmake --install build
}

function main()
{
    parse_args $@

    build_exe
}

main $@
