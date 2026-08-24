from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from rl_course.common import emit_json, positive_int


def run_random_policy(
    episodes: int = 10,
    max_steps: int = 1_000,
    seed: int = 42,
    render: bool = False,
) -> dict[str, float | int]:
    env = gym.make("MountainCarContinuous-v0", render_mode="human" if render else None)
    env.action_space.seed(seed)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    try:
        for episode in range(episodes):
            env.reset(seed=seed if episode == 0 else None)
            total = 0.0
            steps = 0
            for steps in range(1, max_steps + 1):
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                total += float(reward)
                if terminated or truncated:
                    successes += int(terminated)
                    break
            returns.append(total)
            lengths.append(steps)
    finally:
        env.close()
    return {
        "episodes": episodes,
        "seed": seed,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random actions in MountainCarContinuous.")
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--max-steps", type=positive_int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    emit_json(
        run_random_policy(args.episodes, args.max_steps, args.seed, args.render),
        args.output,
    )


if __name__ == "__main__":
    main()

