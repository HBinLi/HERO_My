"""Strict cumulative-percentile gate; ties remain inactive."""


def historical_capability_gate(group_rate: float, *, threshold: float) -> float:
    """Strictly above the cumulative capability threshold; ties (including 1.0) stay off."""
    if group_rate <= threshold:
        return 0.0
    return max(0.0, (group_rate - threshold) / (1.0 - threshold))
