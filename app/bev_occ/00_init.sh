#!/bin/bash
set -e
CUR_DIR=$(cd "$(dirname "$0")"; pwd)
NPU_SAMPLE_ROOT_DIR=${CUR_DIR}/../..

cd "${CUR_DIR}"
source "${CUR_DIR}/env.sh"
source "${NPU_SAMPLE_ROOT_DIR}/script/platform_config.sh"

source "${CUR_DIR}/model_args.sh" "$@"
BEV_OCC_ONNX_NAME=$BEV_MODEL_ONNX

if [[ ! -f "${NPU_SAMPLE_ROOT_DIR}/data/model/bev_occ/${BEV_OCC_ONNX_NAME}" ]]; then
    echo "Missing ONNX: ${NPU_SAMPLE_ROOT_DIR}/data/model/bev_occ/${BEV_OCC_ONNX_NAME}" >&2
    exit 1
fi
mkdir -p tmp_models
cp "${NPU_SAMPLE_ROOT_DIR}/data/model/bev_occ/${BEV_OCC_ONNX_NAME}" \
   "${CUR_DIR}/tmp_models/${BEV_OCC_ONNX_NAME}"

# Copy only companion files with a model-name boundary (g3 must not match g30).
for bev_npz in "${NPU_SAMPLE_ROOT_DIR}/data/model/bev_occ/${BEV_MODEL_NAME}.npz" \
    "${NPU_SAMPLE_ROOT_DIR}/data/model/bev_occ/${BEV_MODEL_NAME}"{.,_,-}*.npz; do
    [[ -f "$bev_npz" ]] || continue
    cp "$bev_npz" "${CUR_DIR}/tmp_models/"
    echo "Prepared: ${CUR_DIR}/tmp_models/$(basename -- "$bev_npz")"
done

cd "${NPU_SAMPLE_ROOT_DIR}/data"
NUSCENES_DATASET=nuscenes
if [ ! -d "${NUSCENES_DATASET}" ]; then
    tar -zxvf "${NUSCENES_DATASET}.tar.gz"
fi
cd "${CUR_DIR}"

echo "Prepared: ${CUR_DIR}/tmp_models/${BEV_OCC_ONNX_NAME}"

# Single-image models use the existing binary calibration dataset; no export NPZ required.
python3 "$CUR_DIR/prepare_calibration.py" --onnx "$BEV_MODEL_NAME" --fixed-only
