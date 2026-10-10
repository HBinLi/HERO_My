"""Task-only trajectory GRPO used by the Vanilla comparison launchers.

Eligibility and reward sourcing exactly match ``run_4b.sh``; only the
group normalization differs. Entropy, token usage and all efficiency signals
are purposely outside this module.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from run_rl._shared.eligibility import (
    sample_has_trainable_tokens,
    sample_is_failed,
    sample_reward,
)

logger = logging.getLogger(__name__)
EPSILON = 1e-6


@dataclass
class ResolutionTrajectory:
    key: tuple[Any, Any]
    sample_indices: list[int] = field(default_factory=list)
    valid_rewards: list[float] = field(default_factory=list)
    failed: bool = False

    @property
    def eligible(self) -> bool:
        return not self.failed and bool(self.valid_rewards)

    @property
    def reward(self) -> float:
        return sum(self.valid_rewards) / len(self.valid_rewards) if self.eligible else 0.0


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Return binary outcome rewards and group-normalized Task advantages."""
    if not samples:
        return [], []
    estimator = str(getattr(args, "advantage_estimator", "grpo"))
    if estimator not in {"grpo", "gspo", "cispo"}:
        raise ValueError(f"Vanilla Task GRPO requires GRPO/GSPO/CISPO, got {estimator!r}")

    trajectories: dict[tuple[Any, Any], ResolutionTrajectory] = {}
    group_keys: dict[Any, list[tuple[Any, Any]]] = {}
    for sample_index, sample in enumerate(samples):
        group_index, key = _trajectory_key(sample, sample_index)
        if key not in trajectories:
            trajectories[key] = ResolutionTrajectory(key=key)
            group_keys.setdefault(group_index, []).append(key)
        trajectory = trajectories[key]
        trajectory.sample_indices.append(sample_index)
        if sample_is_failed(sample):
            trajectory.failed = True
        elif sample_has_trainable_tokens(sample):
            trajectory.valid_rewards.append(sample_reward(args, sample))

    # Re-key both current Slime fields to the same trajectory partition. This
    # reproduces v0.3 group_id for scheduler grouping and loss aggregation.
    for trajectory_id, trajectory in enumerate(trajectories.values()):
        for sample_index in trajectory.sample_indices:
            samples[sample_index].rollout_id = trajectory_id
            samples[sample_index].trajectory_id = trajectory_id
            metadata = _train_metadata(samples[sample_index])
            metadata["hero_efficiency_advantage_spans"] = []
            metadata.pop("hero_outcome_advantage_bound", None)

    raw_rewards = [0.0] * len(samples)
    scalar_advantages = [0.0] * len(samples)
    eligible_count = 0
    for keys in group_keys.values():
        eligible = [trajectories[key] for key in keys if trajectories[key].eligible]
        eligible_count += len(eligible)
        resolution_advantages = _standardize(
            {trajectory.key: trajectory.reward for trajectory in eligible}
        )
        for key in keys:
            trajectory = trajectories[key]
            reward = trajectory.reward
            advantage = resolution_advantages.get(key, 0.0)
            for sample_index in trajectory.sample_indices:
                raw_rewards[sample_index] = reward
                scalar_advantages[sample_index] = advantage

    logger.info(
        "Vanilla Task GRPO: trajectories=%d eligible=%d",
        len(trajectories),
        eligible_count,
    )
    return raw_rewards, scalar_advantages


def compute_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    """Expand each trajectory-level Task advantage over response tokens."""
    estimator = str(getattr(args, "advantage_estimator", "grpo"))
    if estimator not in {"grpo", "gspo", "cispo"}:
        raise ValueError("Vanilla Task advantage requires GRPO/GSPO/CISPO")

    rewards = rollout_data.get("rewards") or []
    kl = rollout_data.get("kl") or []
    response_lengths = rollout_data.get("response_lengths") or []
    total_lengths = rollout_data.get("total_lengths") or []
    if not (len(rewards) == len(kl) == len(response_lengths) == len(total_lengths)):
        raise ValueError("Vanilla Task advantage inputs have inconsistent lengths")

    advantages: list[torch.Tensor] = []
    for reward, template, response_length, total_length in zip(
        rewards, kl, response_lengths, total_lengths, strict=True
    ):
        response_length = int(response_length)
        full = torch.full(
            (response_length,),
            float(reward),
            dtype=torch.float32,
            device=template.device,
        )
        advantages.append(
            _context_parallel_slice(
                full,
                total_length=int(total_length),
                response_length=response_length,
                expected_local_length=int(template.numel()),
            )
        )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]


def _standardize(values: dict[Any, float]) -> dict[Any, float]:
    """Match Gate's population mean/std Task normalization exactly."""
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    if variance <= 0:
        return {key: 0.0 for key in values}
    denominator = math.sqrt(variance) + EPSILON
    return {key: (value - mean) / denominator for key, value in values.items()}


def _hero_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    hero = metadata.get("hero", {}) if isinstance(metadata, dict) else {}
    return hero if isinstance(hero, dict) else {}


def _train_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "train_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.train_metadata = metadata
    return metadata


def _trajectory_key(sample: Any, position: int) -> tuple[Any, tuple[Any, Any]]:
    group_index = _key_value(getattr(sample, "group_index", None), -1)
    trajectory_index = getattr(sample, "trajectory_id", None)
    if trajectory_index is None:
        trajectory_index = getattr(sample, "group_id", None)
    if trajectory_index is None:
        trajectory_index = getattr(sample, "index", None)
    return group_index, (group_index, _key_value(trajectory_index, position))


def _key_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _context_parallel_slice(
    full_response: torch.Tensor,
    *,
    total_length: int,
    response_length: int,
    expected_local_length: int,
) -> torch.Tensor:
    if expected_local_length == response_length:
        return full_response
    from slime.backends.megatron_utils.cp_utils import (  # type: ignore[import-not-found]
        get_logits_and_tokens_offset_with_cp,
    )

    prompt_length = total_length - response_length
    _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    parts: list[torch.Tensor] = []
    for start, end in token_offsets:
        response_start = max(0, start - prompt_length)
        response_end = max(0, end - prompt_length)
        if response_end > response_start:
            parts.append(full_response[response_start:response_end])
    local = torch.cat(parts) if parts else full_response.new_empty((0,))
    if local.numel() != expected_local_length:
        raise ValueError(
            "context-parallel advantage length mismatch: "
            f"expected {expected_local_length}, got {local.numel()}"
        )
    return local
