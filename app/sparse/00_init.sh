#!/usr/bin/env bash
set -eo pipefail
CUR_DIR=$(cd "$(dirname "$0")"; pwd)
NPU_SAMPLE_ROOT_DIR="$CUR_DIR/../.."
cd "$CUR_DIR"
export NPU_TOOLCHAIN_PATH="${NPU_TOOLCHAIN_PATH:-/home/fjm/tong/npu_toolchain}"
source "$NPU_SAMPLE_ROOT_DIR/script/platform_config.sh"
export LD_LIBRARY_PATH="$NPU_TOOLCHAIN_PATH/linux/simulator/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON=/home/fjm/miniconda3/envs/sparsedrive/bin/python
SOURCE=${1:?Usage: bash 00_init.sh /path/to/S6.onnx}
SOURCE_NAME=$(basename "$SOURCE" .onnx)
S6_VARIANT=${S6_VARIANT:-$SOURCE_NAME}
source "$CUR_DIR/run_env.sh"
test -s "$SOURCE"
test -s "${S6_CALIBRATION_DIR:-calibration}/dataset.txt"
S6_RUN_DIR=$(mktemp -d "$CUR_DIR/results/$S6_VARIANT/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export S6_RUN_DIR
cp "$SOURCE" "$S6_RUN_DIR/source.onnx"
sha256sum "$SOURCE" > "$S6_RUN_DIR/source.sha256"
"$PYTHON" - "$SOURCE" "$S6_RUN_DIR" <<'METADATA'
from pathlib import Path
import shutil,sys
source=Path(sys.argv[1]).resolve().with_suffix('.calibration.npz')
destination=Path(sys.argv[2])/'calibration.npz'
if source.exists() and source != destination.resolve():shutil.copy2(source,destination)
METADATA
"$PYTHON" make_config.py
ln -sfn "$(basename "$S6_RUN_DIR")" "$CUR_DIR/results/$S6_VARIANT/latest"
echo "Results: $S6_RUN_DIR"
echo "Use S6_VARIANT=$S6_VARIANT for steps 01–03."
