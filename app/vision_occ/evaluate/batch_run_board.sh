#!/bin/sh

# 要遍历的文件夹路径：
FOLDER_PATH="" # default
BOARD_MODEL_ORTM="" # ortm path
token=""

function batch_execute()
{
    # 遍历文件夹
    for file in "$FOLDER_PATH"/*; do
        if [ -f "$file" ]; then
            echo "FILE: $file"
            token=${file%_*}
            token=${token##*/}
            echo "token: $token"
            mkdir -p $out_path/$token
            ortm_run_model -m ${BOARD_MODEL_ORTM} -i ${file} -o $out_path/$token
        fi
        # return 
    done
}

function main()
{
    FOLDER_PATH=$1
    out_path=$2
    BOARD_MODEL_ORTM=$3
    echo "FOLDER_PATH: $FOLDER_PATH"
    echo "out_path: $out_path"
    echo "BOARD_MODEL_ORTM: $BOARD_MODEL_ORTM"

    batch_execute
}

main $@
