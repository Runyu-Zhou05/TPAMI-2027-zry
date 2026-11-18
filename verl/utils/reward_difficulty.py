"""Group-aware reward statistics for operationalizing GRPO difficulty."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def group_reward_statistics(
    values: Iterable[float],
    *,
    variance_epsilon: float = 1e-12,
    success_threshold: float = 0.5,
    reward_round_decimals: int = 8,
) -> dict:
    """Compute one prompt group's reward diversity and collapse statistics.

    Entropy is the empirical categorical entropy after deterministic rounding.
    For binary rewards this is exactly Bernoulli reward entropy in bits.
    """

    rewards = np.asarray(list(values), dtype=np.float64)
    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError(f"A GRPO reward group needs at least two rollouts, got shape={rewards.shape}")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Reward groups must contain only finite values")
    if variance_epsilon < 0:
        raise ValueError("variance_epsilon must be non-negative")
    if reward_round_decimals < 0:
        raise ValueError("reward_round_decimals must be non-negative")

    rounded = np.round(rewards, decimals=reward_round_decimals)
    _, counts = np.unique(rounded, return_counts=True)
    probabilities = counts.astype(np.float64) / rewards.size
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    # Normalize by the maximum entropy over the reward values observed in this
    # group. For binary rewards this is log2(2)=1; collapsed groups stay zero.
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
    successes = rewards > success_threshold
    variance = float(np.var(rewards, ddof=0))
    all_fail = bool(not np.any(successes))
    all_success = bool(np.all(successes))
    return {
        "all_fail": all_fail,
        "all_success": all_success,
        "group_size": int(rewards.size),
        "informative": bool(variance > variance_epsilon),
        "mixed_success": bool(not all_fail and not all_success),
        "reward_entropy_bits": entropy_bits,
        "reward_entropy_normalized": entropy_bits / max_entropy if max_entropy > 0 else 0.0,
        "reward_max": float(np.max(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_range": float(np.max(rewards) - np.min(rewards)),
        "reward_variance": variance,
        "success_rate": float(np.mean(successes)),
        "unique_reward_count": int(len(counts)),
    }


def aggregate_group_statistics(groups: Iterable[dict]) -> dict:
    """Aggregate prompt-level records without conflating global and within-group variance."""

    records = list(groups)
    if not records:
        raise ValueError("At least one prompt group is required")
    sizes = np.asarray([record["group_size"] for record in records], dtype=np.int64)
    means = np.asarray([record["reward_mean"] for record in records], dtype=np.float64)
    variances = np.asarray([record["reward_variance"] for record in records], dtype=np.float64)
    total = int(np.sum(sizes))
    global_mean = float(np.sum(sizes * means) / total)
    # E[X^2] = E_g[Var(X|g) + E[X|g]^2], weighted by group size.
    global_second_moment = float(np.sum(sizes * (variances + means**2)) / total)
    return {
        "all_fail_group_fraction": float(np.mean([record["all_fail"] for record in records])),
        "all_success_group_fraction": float(np.mean([record["all_success"] for record in records])),
        "global_reward_mean": global_mean,
        "global_reward_std": math.sqrt(max(global_second_moment - global_mean**2, 0.0)),
        "group_size_max": int(np.max(sizes)),
        "group_size_min": int(np.min(sizes)),
        "informative_group_fraction": float(np.mean([record["informative"] for record in records])),
        "mean_group_reward_entropy_bits": float(
            np.mean([record["reward_entropy_bits"] for record in records])
        ),
        "mean_group_reward_entropy_normalized": float(
            np.mean([record["reward_entropy_normalized"] for record in records])
        ),
        "mean_group_reward_variance": float(np.mean(variances)),
        "mixed_success_group_fraction": float(np.mean([record["mixed_success"] for record in records])),
        "prompt_groups": len(records),
        "rollouts": total,
        "success_rate": float(
            np.sum(sizes * np.asarray([record["success_rate"] for record in records])) / total
        ),
    }
