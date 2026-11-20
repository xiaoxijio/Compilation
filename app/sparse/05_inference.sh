#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x build/ortm_run_sim_model ]]; then
  echo 'Run bash 04_build.sh first.' >&2
  exit 2
fi
exec /home/fjm/miniconda3/envs/sparsedrive/bin/python profile_report.py "$@"
