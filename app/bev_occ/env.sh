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
