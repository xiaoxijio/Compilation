#!/usr/bin/env bash
set -eo pipefail
CUR_DIR=$(cd "$(dirname "$0")"; pwd)
NPU_SAMPLE_ROOT_DIR="$CUR_DIR/../.."
cd "$CUR_DIR"
export NPU_TOOLCHAIN_PATH="${NPU_TOOLCHAIN_PATH:-/home/fjm/tong/npu_toolchain}"
source "$NPU_SAMPLE_ROOT_DIR/script/platform_config.sh"
export LD_LIBRARY_PATH="$NPU_TOOLCHAIN_PATH/linux/simulator/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON=/home/fjm/miniconda3/envs/sparsedrive/bin/python
source "$CUR_DIR/run_env.sh"
test -s "$S6_RUN_DIR/model.onnx"
WORK_DIR=$(mktemp -d "$CUR_DIR/.compile.XXXXXX")
trap '"$PYTHON" -c "import os,shutil,sys;shutil.rmtree(sys.argv[1])" "$WORK_DIR"' EXIT
cd "$WORK_DIR"
"$OMG" --model "$S6_RUN_DIR/model.onnx" --config "$S6_RUN_DIR/input_cfg.yml" --output_model "$WORK_DIR/s6.ortm" --target "$TARGET" --log_level info 2>&1 | tee "$S6_RUN_DIR/compile.log"
test -s "$WORK_DIR/s6.ortm"
mv "$WORK_DIR/s6.ortm" "$S6_RUN_DIR/$S6_VARIANT.ortm"
ln -sfn "../results/$S6_VARIANT/$(basename "$S6_RUN_DIR")/$S6_VARIANT.ortm" "$CUR_DIR/models/$S6_VARIANT.ortm"
if [[ -f "$S6_RUN_DIR/calibration.npz" ]]; then
  ln -sfn "../results/$S6_VARIANT/$(basename "$S6_RUN_DIR")/calibration.npz" "$CUR_DIR/models/$S6_VARIANT.calibration.npz"
fi
sha256sum "$S6_RUN_DIR/$S6_VARIANT.ortm" > "$S6_RUN_DIR/ortm.sha256"
rm -- "$S6_RUN_DIR/source.onnx"
echo "Saved $S6_RUN_DIR/$S6_VARIANT.ortm"
