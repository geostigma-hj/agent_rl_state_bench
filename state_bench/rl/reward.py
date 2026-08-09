"""Reward helpers for STATE-Bench customer-support RL."""

from __future__ import annotations

from typing import Any


def compute_reward_v1(
    *,
    state_requirements_met: int | None,
    tool_errors: int,
    redundant_calls: int,
    tool_error_weight: float = 0.05,
    redundant_call_weight: float = 0.0,
    max_tool_error_penalty: float = 0.5,
    max_redundant_penalty: float = 0.0,
    min_reward: float = -1.0,
) -> dict[str, float]:
    """Final-state reward plus conservative tool-quality penalties."""
    state_reward = 1.0 if state_requirements_met == 1 else 0.0
    tool_error_penalty = min(max_tool_error_penalty, tool_error_weight * max(0, tool_errors))
    redundant_penalty = min(max_redundant_penalty, redundant_call_weight * max(0, redundant_calls))
    reward = max(min_reward, state_reward - tool_error_penalty - redundant_penalty)
    return {
        "state_reward": round(state_reward, 4),
        "reward_v1": round(reward, 4),
        "tool_error_penalty": round(tool_error_penalty, 4),
        "redundant_penalty": round(redundant_penalty, 4),
    }


def compute_reward_v1_from_trajectory(traj: dict[str, Any]) -> dict[str, float]:
    return compute_reward_v1(
        state_requirements_met=traj.get("state_requirements_met"),
        tool_errors=int(traj.get("tool_errors") or 0),
        redundant_calls=int(traj.get("redundant_calls") or 0),
    )
