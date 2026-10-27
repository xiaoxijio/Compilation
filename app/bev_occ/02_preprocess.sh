#!/usr/bin/env bash
set -e
CUR_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
source "$CUR_DIR/model_args.sh" "$@"
echo "No preprocessing performed for $BEV_MODEL_NAME."
echo 'This step only works as a placeholder for already-prepared calibration binaries.'
echo 'Dynamic geometry requires calibration data for every model input and a matching YAML.'
