#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
SAMPLE_DIR="$PROJECT_ROOT"
CUSTOM_DIR="$SAMPLE_DIR/app/custom"
TEMPLATE_DIR="$SCRIPT_DIR/template"
TOP_CMAKE_FILE="$SAMPLE_DIR/CMakeLists.txt"

FORCE=0
MODEL_PATH=""

for arg in "$@"; do
    if [ "$arg" = "-f" ] || [ "$arg" = "--force" ]; then
        FORCE=1
    elif [ -f "$arg" ]; then
        MODEL_PATH="$arg"
    else
        echo "Unknown argument: $arg"
        exit 1
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 [-f] <model_path>"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\.[^.]*$//')
TARGET_DIR="$CUSTOM_DIR/$MODEL_NAME"
mkdir -p "$CUSTOM_DIR"

if [ -d "$TARGET_DIR" ]; then
    if [ $FORCE -eq 1 ]; then
        rm -rf "$TARGET_DIR"
    else
        echo "Directory exists: $TARGET_DIR"
        echo "Use -f to force overwrite."
        exit 1
    fi
fi

mkdir -p "$TARGET_DIR"

echo "Generating sample for model: $MODEL_NAME"

for file in "$TEMPLATE_DIR"/*; do
    filename=$(basename "$file")
    dst="$TARGET_DIR/$filename"

    sed "s/{{MODEL_NAME}}/${MODEL_NAME}/g" "$file" > "$dst"
    echo "Generated file: $dst"

    if [[ "$dst" == *.sh ]]; then
        chmod +x "$dst"
    else
        chmod 644 "$dst"
    fi
done

if [[ -f "$TOP_CMAKE_FILE" ]]; then
    SUBDIR_LINE="add_subdirectory(app/custom/$MODEL_NAME)"

    # 删除旧记录
    grep -vF "$SUBDIR_LINE" "$TOP_CMAKE_FILE" > "${TOP_CMAKE_FILE}.tmp"

    # 添加新记录
    echo "$SUBDIR_LINE" >> "${TOP_CMAKE_FILE}.tmp"

    # 覆盖原文件
    mv "${TOP_CMAKE_FILE}.tmp" "$TOP_CMAKE_FILE"
    echo "Updated CMakeLists.txt with: $SUBDIR_LINE"
    else
    echo "Warning: Top-level CMakeLists.txt not found. Skipping update."
fi
