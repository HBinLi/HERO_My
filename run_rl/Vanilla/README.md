# Vanilla: resolution-only GRPO

`train_4b.sh`, `train_9b.sh`, and `train_35b.sh` reproduce the corresponding
original `Vanilla_0821` entries at a 65,536-token context.

- `resolution_advantage.py` computes the binary resolution reward, group-relative
  population normalization, and token-level expansion without efficiency credit.
- `reward.py` applies per-trace `rollout_id` and `trajectory_id` values only after
  trajectory-level GRPO normalization. This preserves the original trace
  scheduling and loss denominator.
- `launch.sh` selects these hooks, disables the second advantage normalization,
  clears inherited efficiency switches, and delegates model/runtime settings to
  `../_shared/run_*.sh`.

Vanilla is intentionally distinct from `prorl_ori`: it uses the restored
trajectory-normalized GRPO reward hook and then per-trace identities, whereas
the original ProRL launchers retain their default Slime bridge hooks.

This method depends only on the common HERO runtime and `run_rl/_shared/`.
No other method directory is required.
