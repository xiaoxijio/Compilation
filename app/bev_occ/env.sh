#!/usr/bin/env bash
# This runner is always PC simulator; leave the global board settings untouched.
BEV_OCC_SIM_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${NPU_TOOLCHAIN_PATH:-} ]]; then
    bev_sdk_root=$BEV_OCC_SIM_DIR
    while [[ ! -d "$bev_sdk_root/npu_toolchain/linux/simulator" ]]; do
        if [[ $bev_sdk_root == / ]]; then
            echo 'Cannot locate SDK. Set NPU_TOOLCHAIN_PATH to its npu_toolchain directory.' >&2
            return 1
        fi
        bev_sdk_root=$(dirname -- "$bev_sdk_root")
    done
    NPU_TOOLCHAIN_PATH=$bev_sdk_root/npu_toolchain
fi
NPU_TOOLCHAIN_PATH=$(cd -- "$NPU_TOOLCHAIN_PATH" && pwd)
export NPU_TOOLCHAIN_PATH
export LD_LIBRARY_PATH="$NPU_TOOLCHAIN_PATH/linux/simulator/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# OMG loads unversioned CUDA libraries; use the installed CUDA 11.8/cuDNN 8 set.
# The system /usr/local/cuda currently points to CUDA 13 without libcudnn.so.
if [[ -f /usr/local/cuda-11.8/lib64/libcudnn.so ]]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi
