#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
source "$script_dir/env.sh"
exec python3 "$script_dir/run_sim.py" "$@"
