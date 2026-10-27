#!/bin/bash
echo "OMG check model"
set -e
CUR_DIR=$(cd "$(dirname "$0")"; pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

cd "${CUR_DIR}"
source "${CUR_DIR}/env.sh"
source "${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh"
source "${CUR_DIR}/model_args.sh" "$@"

[[ -f "./tmp_models/${BEV_MODEL_ONNX}" ]] || { echo "Missing ONNX; run 00_init.sh ${BEV_MODEL_NAME} first" >&2; exit 1; }
[[ -f "$BEV_CONFIG_ARG" ]] || { echo "Missing config: $BEV_CONFIG_ARG" >&2; exit 1; }

if [[ $(check_target "${TARGET}") = "0" ]]; then
    echo "${TARGET} not support"
    exit 1
fi

"${OMG}"                                             \
    --model             "./tmp_models/${BEV_MODEL_ONNX}"    \
    --config            "${BEV_CONFIG_ARG}"               \
    --target            "${TARGET}"                 \
    --check
