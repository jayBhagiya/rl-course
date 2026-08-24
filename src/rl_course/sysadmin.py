from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_course.common import emit_json, positive_int

RUNNING, CRASHED = 0, 1


@dataclass(frozen=True)
class SysadminInstance:
    num_servers: int
    connections: list[tuple[int, int]]
    alpha: float
    eps: float

    def __post_init__(self) -> None:
        if self.num_servers < 2:
            raise ValueError("num_servers must be at least 2")
        if self.alpha < 0 or self.eps < 0 or self.alpha + self.eps > 1:
            raise ValueError("alpha and eps must be non-negative and sum to at most 1")

    @cached_property
    def adjacency(self) -> np.ndarray:
        adjacency = np.zeros((self.num_servers, self.num_servers), dtype=np.int8)
        for left, right in self.connections:
            if left == right or not (0 <= left < self.num_servers and 0 <= right < self.num_servers):
                raise ValueError(f"invalid connection {(left, right)}")
            adjacency[left, right] = adjacency[right, left] = 1
        return adjacency

    @cached_property
    def neighbor_counts(self) -> np.ndarray:
        counts = self.adjacency.sum(axis=0)
        if np.any(counts == 0):
            raise ValueError("every server must have at least one neighbor")
        return counts


DEFAULT_INSTANCE = SysadminInstance(
    num_servers=6,
    connections=[(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)],
    alpha=0.5,
    eps=0.1,
)


def compute_crash_probabilities(
    state: np.ndarray,
    action: int,
    instance: SysadminInstance,
) -> np.ndarray:
    if state.shape != (instance.num_servers,):
        raise ValueError(f"state must have shape {(instance.num_servers,)}")
    if not 0 <= action < instance.num_servers:
        raise ValueError(f"invalid server index {action}")
    crashed_neighbors = state @ instance.adjacency
    probabilities = instance.eps + instance.alpha * crashed_neighbors / instance.neighbor_counts
    probabilities = probabilities.astype(float)
    probabilities[state == CRASHED] = 1.0
    probabilities[action] = 0.0
    return probabilities


class SysadminEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instance: SysadminInstance = DEFAULT_INSTANCE) -> None:
        super().__init__()
        self.instance = instance
        self.action_space = spaces.Discrete(instance.num_servers)
        self.observation_space = spaces.MultiBinary(instance.num_servers)
        self._state = np.zeros(instance.num_servers, dtype=np.int8)

    @property
    def fraction_crashed(self) -> float:
        return float(np.mean(self._state == CRASHED))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state = np.zeros(self.instance.num_servers, dtype=np.int8)
        return self._state.copy(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        probabilities = compute_crash_probabilities(self._state, int(action), self.instance)
        self._state = (self.np_random.random(self.instance.num_servers) < probabilities).astype(np.int8)
        return self._state.copy(), -self.fraction_crashed, False, False, {}


class CrashedFirstPolicy:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: np.ndarray) -> int:
        crashed = np.flatnonzero(state == CRASHED)
        candidates = crashed if len(crashed) else np.arange(len(state))
        return int(self.rng.choice(candidates))


class RandomPolicy:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: np.ndarray) -> int:
        return int(self.rng.integers(len(state)))


def sample_return(
    env: SysadminEnv,
    policy: Callable[[np.ndarray], int],
    horizon: int = 100,
    gamma: float = 1.0,
    seed: int | None = 42,
) -> float:
    state, _ = env.reset(seed=seed)
    total = 0.0
    discount = 1.0
    for _ in range(horizon):
        state, reward, _, _, _ = env.step(policy(state))
        total += discount * reward
        discount *= gamma
    return total


def estimate_policy(
    episodes: int = 100,
    horizon: int = 100,
    gamma: float = 1.0,
    seed: int = 42,
    policy_name: str = "crashed-first",
) -> dict[str, float | int]:
    env = SysadminEnv()
    policy = CrashedFirstPolicy(seed) if policy_name == "crashed-first" else RandomPolicy(seed)
    rng = np.random.default_rng(seed)
    returns = [
        sample_return(env, policy, horizon, gamma, int(rng.integers(0, 2**32)))
        for _ in range(episodes)
    ]
    env.close()
    return {
        "episodes": episodes,
        "horizon": horizon,
        "gamma": gamma,
        "seed": seed,
        "policy": policy_name,
        "mean_return": float(np.mean(returns)),
        "sample_std": float(np.std(returns, ddof=1)) if episodes > 1 else 0.0,
        "standard_error": float(np.std(returns, ddof=1) / np.sqrt(episodes)) if episodes > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate maintenance policy in Sysadmin MDP.")
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--horizon", type=positive_int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--policy", choices=("random", "crashed-first"), default="crashed-first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be in [0, 1]")
    emit_json(
        estimate_policy(args.episodes, args.horizon, args.gamma, args.seed, args.policy),
        args.output,
    )


if __name__ == "__main__":
    main()
