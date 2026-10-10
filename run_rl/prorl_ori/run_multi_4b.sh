#!/usr/bin/env bash
# Original ProRL recipe with the unchanged shared backend defaults.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_METHOD=prorl_ori
exec bash "${SCRIPT_DIR}/../_shared/run_4b.sh" "$@"
