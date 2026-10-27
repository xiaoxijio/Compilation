#!/bin/bash

function prepar_llm_model {
    local model_name="$1"

    case "$model_name" in
        "qwen")
            model_path="model/qwen/model.safetensors"
            target_dir="${NPU_SAMPLE_ROOT_DIR}/data/llm/qwen"
            ;;
        "tinyllama")
            model_path="model/tinyllama/model.safetensors"
            target_dir="${NPU_SAMPLE_ROOT_DIR}/data/llm/tinyllama"
            ;;
        *)
            echo "Error: only supported qwen and tinyllama"
            return 1
            ;;
    esac

    if [[ ! -f "$target_dir/model.safetensors" ]]; then
        echo "Preparing LLM model: $model_name ..."
        mkdir -p "$target_dir"
        fetch_evaluation_datasets "$model_path" "$target_dir"
        echo "$model_name downloaded to $target_dir"
    else
        echo "$model_name is already prepared at $target_dir"
    fi
}