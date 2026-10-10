#!/usr/bin/env bash
# Shared 64K Vanilla GRPO launch configuration with per-trace scheduling/loss.

scaling_launch() {
  : "${SCALING_MODEL_NAME:?SCALING_MODEL_NAME is required}"
  : "${SCALING_LAUNCHER:?SCALING_LAUNCHER is required}"
  : "${SCALING_EXP_TAG:?SCALING_EXP_TAG is required}"
  : "${SCALING_METHOD:?SCALING_METHOD is required}"

  local common_dir repo_root run_date stamp
  common_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="${REPO_ROOT:-$(cd "${common_dir}/../.." && pwd)}"
  run_date="${RUN_DATE:-$(date +%m%d)}"
  stamp="${STAMP:-$(date +%Y%m%d_%H%M%S)_${BASHPID}}"

  [[ -f "${repo_root}/${SCALING_LAUNCHER}" ]] || {
    echo "Missing scaling launcher: ${repo_root}/${SCALING_LAUNCHER}" >&2
    return 1
  }

  export EXP_TAG="${EXP_TAG:-${SCALING_EXP_TAG}}"
  export WANDB_GROUP="${WANDB_GROUP:-${SCALING_MODEL_NAME}_${EXP_TAG}}"
  export RUN_DATE="${run_date}"
  export STAMP="${stamp}"
  export RESULTS_ROOT="${RESULTS_ROOT:-${repo_root}/../results}"
  export RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/train/Vanilla/${SCALING_MODEL_NAME}/${stamp}_${EXP_TAG}}"
  [[ "${RUN_ROOT}" == /* ]] || RUN_ROOT="${PWD}/${RUN_ROOT}"
  export RUN_ID="${RUN_ID:-${SCALING_MODEL_NAME}_${EXP_TAG}_${stamp}}"

  unset CLAUDE_CODE_MAX_TURNS RUN_MULTI_PRESERVE_MAX_TURNS
  export SGLANG_CONTEXT_LENGTH=65536

  # Clear any inherited loss-normalization override before selecting the method.
  unset SLIME_LOSS_NORMALIZATION_BY_TRAJECTORY_ID

  case "${SCALING_METHOD}" in
    vanilla)
      # Keep standard Vanilla Task GRPO, then reproduce the 2026-08-21
      # trace-level scheduling and loss identities as an isolated ablation.
      export CUSTOM_REWARD_POST_PROCESS_PATH="run_rl.Vanilla.reward.post_process_rewards"
      export CUSTOM_ADVANTAGE_FUNCTION_PATH="run_rl.Vanilla.resolution_advantage.compute_advantages"
      unset CUSTOM_ROLLOUT_LOG_FUNCTION_PATH
      export SLIME_NORMALIZE_ADVANTAGES=0
      unset HERO_GENERATION_ENTROPY_TOP_K
      unset EFFICIENCY_COEFFICIENT GATE_THRESHOLD COARSE_SUCCESS_ONLY
      unset PRORL_TASK_ADV PRORL_COARSE_VARIANT
      ;;
    *) echo "Unknown SCALING_METHOD=${SCALING_METHOD}" >&2; return 2 ;;
  esac

  unset HERO_FAILURE_TOKEN_EFFICIENCY_ENABLED HERO_FAILURE_TOKEN_EFFICIENCY_CREDIT

  echo "[scaling] model=${SCALING_MODEL_NAME} method=${SCALING_METHOD} experiment=${EXP_TAG}"
  echo "[scaling] output=${RUN_ROOT}"
  exec bash "${repo_root}/${SCALING_LAUNCHER}" "$@"
}
