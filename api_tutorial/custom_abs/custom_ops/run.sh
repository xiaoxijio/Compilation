#!/bin/bash

export LD_LIBRARY_PATH=../../lib

set -e

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR=${CURRENT_DIR}/build
BUILD_TYPE="-DCMAKE_BUILD_TYPE=Debug"
ENABLE_BOARD_V100="-DENABLE_BOARD_V100=OFF"
ENABLE_BOARD_V200="-DENABLE_BOARD_V200=OFF"

function build_clean()
{
    rm -rf ${BUILD_DIR}
}

function parse_args()
{
    for arg in $@
    do
        if [ "X$arg" = "Xclean" ] || [ "X$arg" = "XCLEAN" ]; then
            build_clean
            echo "clean up"
            exit 0
        elif [ "X$arg" = "Xdebug" ] || [ "X$arg" = "XDEBUG" ]; then
            BUILD_TYPE="-DCMAKE_BUILD_TYPE=Debug"
            echo "Build mode debug"
        elif [ "X$arg" = "Xrelease" ] || [ "X$arg" = "XRELEASE" ]; then
            BUILD_TYPE="-DCMAKE_BUILD_TYPE=Release"
            echo "Build mode release"
        elif [ "X$arg" = "Xboard_v100" ] || [ "X$arg" = "XBOARD_V100" ]; then
            ENABLE_BOARD_V100="-DENABLE_BOARD_V100=ON"
            echo "Build board v100 custom_opsets_extern"
        elif [ "X$arg" = "Xboard_v200" ] || [ "X$arg" = "XBOARD_V200" ]; then
            ENABLE_BOARD_V200="-DENABLE_BOARD_V200=ON"
            echo "Build board v200 custom_opsets_extern"
        else
            echo "$arg is an unkonwn option, only support ./build.sh [release/debug]"
            exit -1
        fi
    done
}

function main()
{
    parse_args $@
    cmake -B ${BUILD_DIR} ${BUILD_TYPE} ${ENABLE_BOARD_V100} ${ENABLE_BOARD_V200}
    cmake --build ${BUILD_DIR} -j32
}

main $@
