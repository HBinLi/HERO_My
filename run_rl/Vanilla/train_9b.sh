#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/launch.sh"
SCALING_MODEL_NAME=qwen35_9b
SCALING_LAUNCHER=run_rl/_shared/run_9b.sh
SCALING_EXP_TAG=Vanilla_context64k_trace_trace
SCALING_METHOD=vanilla
scaling_launch "$@"
