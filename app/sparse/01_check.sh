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
MODEL="$S6_RUN_DIR/source.onnx"
LOG_NAME=check_source.log
CSV_NAME=source_operators.csv
WORK_DIR=$(mktemp -d "$CUR_DIR/.check.XXXXXX")
trap '"$PYTHON" -c "import os,shutil,sys;shutil.rmtree(sys.argv[1])" "$WORK_DIR"' EXIT
cd "$WORK_DIR"
"$OMG" --model "$MODEL" --config "$S6_RUN_DIR/input_cfg.yml" --target "$TARGET" --check 2>&1 | tee "$S6_RUN_DIR/$LOG_NAME"
"$PYTHON" - "$WORK_DIR/check_results.csv" "$S6_RUN_DIR/$CSV_NAME" <<'CHECK'
import csv,sys,shutil
from pathlib import Path
p=Path(sys.argv[1]);rows=list(csv.DictReader(p.open()))
assert rows and all(r['Supportability']=='Supported' and r['HardWare']=='NPU' for r in rows), 'S6 requires all operators on NPU'
shutil.copy2(p,sys.argv[2])
print('Verified',len(rows),'NPU operators')
CHECK
