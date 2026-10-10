#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/launch.sh"
SCALING_MODEL_NAME=qwen35_9b
SCALING_LAUNCHER=run_rl/_shared/run_9b.sh
ratio_launch "$@"
