#!/bin/sh

# 要遍历的文件夹路径：
FOLDER_PATH="" # default
BOARD_MODEL_ORTM=$3 # ortm path

function batch_execute()
{
    # 遍历文件夹
    for file in "$FOLDER_PATH"/*; do
        if [ -f "$file" ]; then
            echo "FILE: $file"
        elif [ -d "$file" ]; then
            echo "FOLDER: $file"
            cd $file
            # chmod 777 -R ./ 
            mkdir -p npu_out
            input_file_name=$(ls *_6x3x256x704.bin)
            ortm_run_model -m ${BOARD_MODEL_ORTM} -i ${input_file_name} -o npu_out
            cd -
        fi
        # return 
    done
}

function help()
{
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i: sample_tokens的路径"
    echo "  -h, --help  Show this help message"
}

function parse_args()
{
    ARGS=$(getopt -o hi: -n"$0" -- "$@")
    eval set -- "$ARGS"
    while [ -n "$1" ]; do
        case $1 in
        -h) help; exit 0 ;;
        -i) FOLDER_PATH=$2 ;;
        --) ;;
        *)
            echo "$1 is an unkonwn option"
            shift
            ;;
        esac
        shift
    done
}

function main()
{
    parse_args $@

    batch_execute
}

main $@
