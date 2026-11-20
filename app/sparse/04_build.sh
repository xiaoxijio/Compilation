#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
export NPU_TOOLCHAIN_PATH="/home/fjm/tong/npu_toolchain"
cmake -S "$script_dir" -B "$script_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DNPU_TOOLCHAIN_PATH="$NPU_TOOLCHAIN_PATH"
cmake --build "$script_dir/build" --parallel "${JOBS:-4}"
echo "Built PC simulator: $script_dir/build/ortm_run_sim_model"
