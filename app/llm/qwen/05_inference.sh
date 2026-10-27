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
    EXPORT_LIB_PATH="${NPU_SAMPLE_ROOT_DIR}/third_party/sentencepiece_lib/aarch64/lib"
    case "$LD_LIBRARY_PATH" in
        "*sentencepiece*")   ;;
        *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${EXPORT_LIB_PATH}" ;;
    esac
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
    LLM=sample_qwen

    ./${LLM}  \
    --tokenizer_type            1 \
    --decoder_filename          "./tmp_models/qwen/decoder_%d.ortm" \
    --filename_post_model       "./tmp_models/qwen/post_model.ortm" \
    --filename_tokens_embed     "./tmp_res/DeepSeek-R1-Distill-Qwen-1.5B-fp16.bin" \
    --filename_tokenizer_model  "./tmp_res/tokenizer.model" \
    --prompt ""
}

function main()
{
    inference $@
}

main $@
