# Original ProRL

These entries select the original ProRL recipe through the common execution
infrastructure. They do not import another method directory.

```bash
WANDB_MODE=disabled bash run_rl/prorl_ori/run_multi_4b.sh
# Alternatives: run_multi_9b.sh or run_multi_35b_a3b.sh.
```

Prepare the base repository using `Prepare.md`. The launchers use
`run_rl/_shared/run_*.sh`, which retain the original model and runtime
defaults. The default reward post-processor remains
`slime_bridge.reward_post_process.post_process_rewards`; advantage construction
and normalization retain the original backend defaults. Results are written to
`../results/train/prorl_ori/` unless overridden.
