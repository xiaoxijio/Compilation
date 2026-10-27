#!/bin/bash

# 检查是否传入了路径参数
if [ -z "$1" ]; then
    echo "Please set sample path"
    echo "usage: $0 <PATH>"
    exit 1
fi

SCRIPT_DIR="$1"

# check path
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "get sample path failed: '$SCRIPT_DIR' is invalid."
    exit 1
fi

# shell list
scripts=("00_init.sh" "01_check.sh" "02_preprocess.sh" "03_omg.sh" "04_build.sh" "05_inference.sh")

# run shell
for script in "${scripts[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        echo "Start Running: $script_path"
        bash "$script_path"
        if [ $? -eq 0 ]; then
            echo "$script success"
        else
            echo "$script fail"
            exit 1
        fi
    else
        echo "Warning: '$script_path' is not exist or no permission"
        continue
    fi
done
