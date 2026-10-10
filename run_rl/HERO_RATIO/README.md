# HERO_RATIO: configurable cumulative capability gate

This variant uses the same resolution, coarse/fine efficiency credit and
resolution-first clipping as HERO. The gate threshold comes from the cumulative
history of eligible group resolution rates instead of a fixed threshold.

## Launch

```bash
# Default ratio: 40%, efficiency coefficient: 0.3.
bash run_rl/HERO_RATIO/train_4b.sh

# Upper 30%; the historical threshold is P70.
EFFICIENCY_RATIO=30 bash run_rl/HERO_RATIO/train_9b.sh

# Decimal percentages and coefficient overrides are supported.
EFFICIENCY_RATIO=25.5 EFFICIENCY_COEFFICIENT=0.2 bash run_rl/HERO_RATIO/train_35b.sh
```

`EFFICIENCY_RATIO` is a percentage in [0, 100], not a fraction: `40` means 40%.
Decimal and scientific notation are accepted (for example, `.5` means 0.5%).
Its default is 40 for all three models. The cumulative quantile is
`1 - EFFICIENCY_RATIO / 100`; current accepted-batch observations are included.
Only groups strictly above the resulting threshold activate efficiency credit.
Ties remain inactive, so the actual active fraction is not guaranteed to equal
the requested percentage. At 0 the gate is always off; at 100 groups tied at
the historical minimum still remain off. Existing group eligibility and minimum
sample requirements also remain in effect.

The entry scripts clear inherited `GATE_THRESHOLD`; that variable controls the
fixed-threshold HERO variant. `EFFICIENCY_COEFFICIENT` defaults to 0.3 here.
Both ratio and coefficient are explicitly forwarded to Ray jobs and nested
actors. Default experiment tags include the selected ratio and coefficient.

## Modules

- `hierarchical_policy_optimization.py`: ratio variant reward/advantage hooks.
- `capability_history.py`: cumulative rates, percentile calculation and resume state.
- `efficiency_gating.py`: strict historical-threshold gate.
- `launch.sh`, `train_{4b,9b,35b}.sh`: model entry points.

Shared paper-aligned credit, clipping and token-span modules are imported from
this directory. The calculations are copied unchanged from the original
implementation; no `run_rl.HERO` import is required.

## Checkpoint compatibility

New runs persist `runtime/gate_ratio_history.json` beside the checkpoint
directory, with schema 2 and an explicit `ratio_percent`. Resume requires the
same ratio; choose a new run directory to change it. An in-process ratio change
is rejected as well.

Legacy schema-1 `gate_ratio40_history.json` files remain readable at ratio 40;
the next write uses the generic schema-2 file. If both files exist, the generic
file takes precedence. Keep the history alongside the saved/loaded checkpoint.
Default 40% reward, token-advantage and trace-identity calculations preserve the
previous ratio40 behavior; the new state filename/schema is intentional.

Launchers use `run_rl/_shared/` for model and runtime setup. No other method
directory is required.
