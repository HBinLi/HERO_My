#!/usr/bin/env bash
# HERO cumulative-capability-history variant; independent of fixed-gate settings.
ratio_launch() {
  local ratio_dir repo_root run_date stamp
  ratio_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="${REPO_ROOT:-$(cd "${ratio_dir}/../.." && pwd)}"
  run_date="${RUN_DATE:-$(date +%m%d)}"
  stamp="${STAMP:-$(date +%Y%m%d_%H%M%S)_${BASHPID}}"
  : "${SCALING_MODEL_NAME:?}"
  : "${SCALING_LAUNCHER:?}"
  [[ -f "${repo_root}/${SCALING_LAUNCHER}" ]] || return 1
  export EFFICIENCY_COEFFICIENT="${EFFICIENCY_COEFFICIENT:-0.3}"
  local ratio
  ratio="$(LC_ALL=C awk -v value="${EFFICIENCY_RATIO:-40}" 'BEGIN {
    if (value !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ || value + 0 < 0 || value + 0 > 100) exit 1;
    printf "%.15g", value + 0
  }')" || { echo "EFFICIENCY_RATIO must be a percentage in [0, 100]" >&2; return 2; }
  export EFFICIENCY_RATIO="${ratio}"
  export EXP_TAG="${EXP_TAG:-HERO_RATIO_context64k_lambda${EFFICIENCY_COEFFICIENT}_ratio${EFFICIENCY_RATIO}}"
  export WANDB_GROUP="${WANDB_GROUP:-${SCALING_MODEL_NAME}_${EXP_TAG}}"
  export RUN_DATE="${run_date}" STAMP="${stamp}"
  export RESULTS_ROOT="${RESULTS_ROOT:-${repo_root}/../results}"
  export RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/train/HERO_RATIO/${SCALING_MODEL_NAME}/${stamp}_${EXP_TAG}}"
  [[ "${RUN_ROOT}" == /* ]] || RUN_ROOT="${PWD}/${RUN_ROOT}"
  export RUN_ID="${RUN_ID:-${SCALING_MODEL_NAME}_${EXP_TAG}_${stamp}}"

  unset CLAUDE_CODE_MAX_TURNS RUN_MULTI_PRESERVE_MAX_TURNS
  unset SLIME_LOSS_NORMALIZATION_BY_TRAJECTORY_ID
  unset GATE_THRESHOLD COARSE_SUCCESS_ONLY PRORL_TASK_ADV PRORL_COARSE_VARIANT
  unset HERO_FAILURE_TOKEN_EFFICIENCY_ENABLED HERO_FAILURE_TOKEN_EFFICIENCY_CREDIT
  unset CUSTOM_ROLLOUT_LOG_FUNCTION_PATH
  export SGLANG_CONTEXT_LENGTH=65536
  export CUSTOM_REWARD_POST_PROCESS_PATH="run_rl.HERO_RATIO.hierarchical_policy_optimization.post_process_rewards"
  export CUSTOM_ADVANTAGE_FUNCTION_PATH="run_rl.HERO_RATIO.hierarchical_policy_optimization.compute_advantages"
  export SLIME_NORMALIZE_ADVANTAGES=0
  export HERO_GENERATION_ENTROPY_TOP_K=10
  echo "[HERO_RATIO] model=${SCALING_MODEL_NAME} experiment=${EXP_TAG}"
  echo "[HERO_RATIO] ratio_percent=${EFFICIENCY_RATIO} gate=strictly-above-cumulative-percentile output=${RUN_ROOT}"
  exec bash "${repo_root}/${SCALING_LAUNCHER}" "$@"
}
