"""Standard Vanilla GRPO followed by legacy per-trace identities.

Reward normalization and token-level advantage construction remain owned by
``run_rl.Vanilla.resolution_advantage``.  This wrapper changes only the
two identities consumed by Slime after reward post-processing:

* ``rollout_id`` controls training-step scheduling;
* ``trajectory_id`` controls the loss denominator.

Assigning a unique value to both for every flattened trace reproduces the
trace/trace behavior of the 2026-08-21 Vanilla64k run without changing the
standard GRPO calculation.
"""

from __future__ import annotations

import logging
from typing import Any

from run_rl.Vanilla.resolution_advantage import (
    post_process_rewards as _standard_grpo_post_process_rewards,
)

logger = logging.getLogger(__name__)


def post_process_rewards(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Run standard Vanilla GRPO, then switch scheduling/loss to traces."""
    raw_rewards, advantages = _standard_grpo_post_process_rewards(args, samples)

    # This must happen after standard GRPO has grouped sibling traces into
    # trajectories and computed the trajectory-level normalized advantages.
    for trace_id, sample in enumerate(samples):
        sample.rollout_id = trace_id
        sample.trajectory_id = trace_id

    logger.info(
        "Vanilla: standard GRPO -> trace scheduling/loss (%d traces)",
        len(samples),
    )
    return raw_rewards, advantages
