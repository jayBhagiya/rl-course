from __future__ import annotations

import argparse
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from rl_course.common import emit_json, positive_int

State = tuple[int, int, int]
Policy = Callable[[State, np.random.Generator], int]


def basic_policy(state: State, _: np.random.Generator) -> int:
    """Course heuristic: return 0 to stick and 1 to hit."""
    player, dealer, usable_ace = state
    if usable_ace:
        if player >= 19:
            return 0
        if player <= 17:
            return 1
        return 1 if dealer in {1, 9, 10} else 0
    if dealer in {1, 7, 8, 9, 10}:
        return int(player <= 16)
    if dealer in {4, 5, 6}:
        return int(player <= 11)
    return int(player <= 12)


def random_policy(_: State, rng: np.random.Generator) -> int:
    return int(rng.integers(2))


def evaluate(policy: Policy, episodes: int = 10_000, seed: int = 42) -> dict[str, float | int]:
    env = gym.make("Blackjack-v1")
    rng = np.random.default_rng(seed)
    outcomes = {-1: 0, 0: 0, 1: 0}
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            while True:
                state, reward, terminated, truncated, _ = env.step(policy(state, rng))
                if terminated or truncated:
                    outcomes[int(reward)] += 1
                    break
    finally:
        env.close()
    return {
        "episodes": episodes,
        "seed": seed,
        "wins": outcomes[1],
        "losses": outcomes[-1],
        "ties": outcomes[0],
        "win_rate": outcomes[1] / episodes,
        "mean_return": (outcomes[1] - outcomes[-1]) / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random or heuristic Blackjack policy.")
    parser.add_argument("--policy", choices=("random", "basic"), default="basic")
    parser.add_argument("--episodes", type=positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    policy = basic_policy if args.policy == "basic" else random_policy
    result = evaluate(policy, args.episodes, args.seed)
    result["policy"] = args.policy
    emit_json(result, args.output)


if __name__ == "__main__":
    main()

