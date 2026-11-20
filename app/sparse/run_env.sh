#!/usr/bin/env bash
S6_VARIANT=${S6_VARIANT:-s6}
if [[ ! "$S6_VARIANT" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ || "$S6_VARIANT" == "profile" ]]; then
  echo "Invalid model name: $S6_VARIANT" >&2
  exit 2
fi
export S6_VARIANT
mkdir -p "$CUR_DIR/models" "$CUR_DIR/results/$S6_VARIANT"
if [[ "$(basename "$0")" != 00_init.sh ]]; then
  S6_RUN_DIR=$(realpath -e "${S6_RUN_DIR:-$CUR_DIR/results/$S6_VARIANT/latest}") || {
    echo "Run 00_init.sh for $S6_VARIANT first." >&2
    exit 2
  }
  export S6_RUN_DIR
fi
