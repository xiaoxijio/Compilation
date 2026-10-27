#!/bin/bash

set -e
CUR_DIR=$(cd $(dirname $0); pwd)

BUILD_TYPE=Release
SAMPLE_BUILD_TARGET="sim"

JOB_NUM=8
ENABLE_OT_AVP_NPU_DEBUG=OFF

show_help() {
    USAGE=$(
		cat <<-END
			$0 -h\t#show this helper
			options:
			\t-c clean. Remove all build files.
			\tdebug     Build debug versions.
			\trelease   Build release versions
			\tboard     Build board versions
			\tsim       Build sim versions
			\tsdk       Build sdk versions
		END
	)
    echo -e "${USAGE}"
}

function build()
{
    if [ ${SAMPLE_BUILD_TARGET} == "board" ]; then
        cross_file="cmake/cross_compile_aarch64_linux.cmake"
    fi
    cd ${CUR_DIR}

    cmake -B build                                           \
        -DCMAKE_TOOLCHAIN_FILE=$cross_file                   \
        -DENABLE_OT_AVP_NPU_DEBUG=${ENABLE_OT_AVP_NPU_DEBUG} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                     \
        -DCMAKE_INSTALL_PREFIX=${CUR_DIR}/output             \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF ${sdk_option}    \
        -DSAMPLE_BUILD_TARGET=${SAMPLE_BUILD_TARGET}
    cmake --build build -j ${JOB_NUM}
    cmake --install build

}

function clean()
{
    cd ${CUR_DIR}

    rm -rf build output
    find ./api_tutorial -name  "*build.sh" -exec bash {} clean \;
    find ./app -name  "*build.sh" -exec bash {} clean \;
    find ./avp_server -name  "*build.sh" -exec bash {} clean \;
}

function parse_args()
{
    ARGS=$(getopt -o aj:v:ch --long clean,debug,release,board,sdk,sim -n"$0" -- "$@")
    eval set -- "$ARGS"
    while [ -n "$1" ]; do
        case $1 in
        -h | \?)
            show_help
            exit 0
            ;;
        -c | clean)
            clean
            echo "clean up"
            exit 0
            ;;
        -j)
            JOB_NUM=$2
            echo "JOB_NUM: $JOB_NUM"
            shift 1
            ;;
        debug)
            echo "build debug"
            BUILD_TYPE=Debug
            ENABLE_OT_AVP_NPU_DEBUG=ON
            ;;
        release)
            BUILD_TYPE=Release
            echo "build release"
            ;;
        board)
            SAMPLE_BUILD_TARGET="board"
            echo "enable board"
            ;;
        sim)
            SAMPLE_BUILD_TARGET="sim"
            echo "enable sim"
            ;;
        sdk)
            SAMPLE_BUILD_TARGET="sdk"
            echo "enable sdk"
            ;;
        --) ;;
        *)
            echo "'$1' is an unkonwn option, only support ./build.sh [clean/debug/release/board/sdk/sim]"
            exit 1
            shift
            ;;
        esac
        shift
    done
}

function main()
{
    parse_args $@
    build
    return 0
}

main $@
