#!/usr/bin/env bash
# Shared arguments for model preparation/check/compilation. No global selection.
BEV_MODEL_ARG=g3
BEV_CONFIG_ARG=input_cfg/input_cfg.yml
bev_model_seen=0
bev_config_seen=0
while (($#)); do
    case "$1" in
        --model-name)
            [[ $# -ge 2 ]] || { echo '--model-name requires a name' >&2; exit 2; }
            BEV_MODEL_ARG=$2; bev_model_seen=1; shift 2 ;;
        --config)
            [[ $# -ge 2 ]] || { echo '--config requires a YAML path' >&2; exit 2; }
            BEV_CONFIG_ARG=$2; bev_config_seen=1; shift 2 ;;
        -h|--help)
            echo "Usage: bash $(basename -- "$0") [MODEL_NAME[.onnx]] [--config FILE.yml]"
            echo 'Default model: g3. Model configs are selected automatically; --config overrides.'
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            [[ $bev_model_seen == 0 ]] || { echo "Unexpected argument: $1" >&2; exit 2; }
            BEV_MODEL_ARG=$1; bev_model_seen=1; shift ;;
    esac
done
BEV_MODEL_NAME=${BEV_MODEL_ARG%.onnx}
if [[ ! $BEV_MODEL_NAME =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || $BEV_MODEL_NAME == *.ortm ]]; then
    echo 'Use a model basename such as bev_occ_mapping or bev_occ_mapping.onnx, not a path.' >&2
    exit 2
fi
BEV_MODEL_ONNX=$BEV_MODEL_NAME.onnx
BEV_MODEL_ORTM=$BEV_MODEL_NAME.ortm

if [[ $bev_config_seen == 0 ]]; then
    BEV_CONFIG_ARG="input_cfg/${BEV_MODEL_NAME}.yml"
fi

# Bare config filenames are resolved in the shared input_cfg directory.
if [[ "$BEV_CONFIG_ARG" != */* ]]; then
    BEV_CONFIG_ARG="input_cfg/$BEV_CONFIG_ARG"
fi
