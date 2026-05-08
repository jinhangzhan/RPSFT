#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD=iw
exec bash "${SCRIPT_DIR}/run_math_sft.sbatch.sh" "$@"
