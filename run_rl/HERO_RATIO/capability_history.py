"""Per-run, replayable cumulative percentile gates; no process-global history."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

STATE_NAME = "gate_ratio_history.json"
LEGACY_STATE_NAME = "gate_ratio40_history.json"


def efficiency_ratio_percent():
    """Target fraction above the capability quantile, expressed as a percentage."""
    value = os.environ.get("EFFICIENCY_RATIO", "40")
    try:
        ratio = float(value)
    except ValueError as exc:
        raise ValueError("EFFICIENCY_RATIO must be a percentage in [0, 100]") from exc
    if not math.isfinite(ratio) or not 0 <= ratio <= 100:
        raise ValueError("EFFICIENCY_RATIO must be a percentage in [0, 100]")
    return ratio


def _linear_quantile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _state_path(checkpoint_dir):
    return Path(checkpoint_dir).parent / "runtime" / STATE_NAME


def prepare_history(args, samples, trajectories, group_keys):
    # Accepted step IDs survive trajectory/trace identity rewrites and refer
    # only to batches delivered to training, not prefetched/rejected groups.
    steps = {
        (sample.metadata or {}).get("hero", {}).get("accepted_rollout_id") for sample in samples
    }
    if len(steps) != 1 or None in steps:
        raise ValueError("ratio requires one accepted_rollout_id for the whole batch")
    step = steps.pop()
    if not isinstance(step, int) or step < 0:
        raise ValueError("ratio accepted_rollout_id must be a nonnegative integer")

    ratio = efficiency_ratio_percent()
    previous_ratio = getattr(args, "_hero_ratio_percent", ratio)
    if previous_ratio != ratio:
        raise ValueError("EFFICIENCY_RATIO cannot change within a run")
    previous = getattr(args, "_hero_ratio_history", None)
    if previous is None:
        previous = {}
        candidates = [
            Path(value).parent / "runtime" / name
            for value in (getattr(args, "save", None), getattr(args, "load", None))
            if value
            for name in (STATE_NAME, LEGACY_STATE_NAME)
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is not None:
            data = json.loads(source.read_text())
            if data.get("schema") not in {1, 2}:
                raise ValueError(f"Unsupported ratio history: {source}")
            stored_ratio = 40.0 if data["schema"] == 1 else data.get("ratio_percent")
            if stored_ratio != ratio:
                raise ValueError(
                    f"EFFICIENCY_RATIO={ratio} differs from saved ratio {stored_ratio}; "
                    "use a new run directory for a different ratio"
                )
            previous = data["batches"]
            for key, batch in previous.items():
                if int(key) < 0 or any(
                    not isinstance(rate, (int, float))
                    or not math.isfinite(rate)
                    or not 0 <= rate <= 1
                    for rate in batch["rates"]
                ):
                    raise ValueError(f"Invalid ratio history: {source}")
        elif step > 1:
            raise ValueError(
                "Resuming ratio requires runtime/gate_ratio_history.json "
                "beside the output or loaded checkpoint directory"
            )

    # Drop replayed/future batches when retrying or resuming an older ckpt.
    history = {key: value for key, value in previous.items() if int(key) < step}
    rates = []
    for keys in group_keys.values():
        completed = [trajectories[key] for key in keys if trajectories[key].efficiency_eligible]
        if completed:
            rates.append(sum(item.resolved for item in completed) / len(completed))
    all_rates = [rate for batch in history.values() for rate in batch["rates"]] + rates
    # Quantile 1 - ratio/100 leaves the requested upper fraction above the gate.
    # Linear interpolation matches the quantile convention used by reward.py.
    # No observations means no group can pass the strict gate.
    threshold = _linear_quantile(all_rates, 1.0 - ratio / 100.0) if all_rates else 1.0
    history[str(step)] = {"rates": rates, "threshold": threshold, "groups": len(group_keys)}
    args._hero_ratio_percent = ratio
    return history, threshold


def commit_history(args, history):
    if getattr(args, "save", None):
        path = _state_path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                name = stream.name
                json.dump(
                    {"schema": 2, "ratio_percent": args._hero_ratio_percent, "batches": history},
                    stream,
                    allow_nan=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)
    args._hero_ratio_history = history
