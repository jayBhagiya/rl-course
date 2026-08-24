from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_course.common import emit_json, positive_int

Transition = tuple[int, int, float]


def first_visit_returns(episode: list[Transition], gamma: float) -> list[tuple[int, int, float]]:
    """Return each state-action pair's first-visit return."""
    returns = [0.0] * len(episode)
    total = 0.0
    for index in range(len(episode) - 1, -1, -1):
        total = episode[index][2] + gamma * total
        returns[index] = total
    first_visits: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for (state, action, _), total in zip(episode, returns):
        if (state, action) not in seen:
            seen.add((state, action))
            first_visits.append((state, action, total))
    return first_visits


def evaluate_policy(
    env: gym.Env,
    policy: np.ndarray,
    episodes: int,
    seed: int,
) -> float:
    successes = 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        while True:
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            if terminated or truncated:
                successes += int(reward > 0)
                break
    return successes / episodes


def monte_carlo_control(
    iterations: int = 50,
    episodes_per_iteration: int = 100,
    gamma: float = 0.9,
    epsilon_start: float = 0.5,
    epsilon_min: float = 0.1,
    epsilon_decay: float = 0.01,
    convergence_tolerance: float | None = None,
    min_iterations: int = 1,
    seed: int = 42,
    slippery: bool = False,
) -> dict:
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    num_states = int(env.observation_space.n)
    num_actions = int(env.action_space.n)
    policy = rng.integers(num_actions, size=num_states)
    q_values = np.zeros((num_states, num_actions), dtype=float)
    visit_counts = np.zeros((num_states, num_actions), dtype=np.int64)
    epsilon = epsilon_start
    metrics: list[dict[str, float | int]] = []
    converged = False
    try:
        for iteration in range(iterations):
            previous = q_values.copy()
            successes = 0
            for episode_index in range(episodes_per_iteration):
                first_reset = iteration == 0 and episode_index == 0
                state, _ = env.reset(seed=seed if first_reset else None)
                episode: list[Transition] = []
                while True:
                    if rng.random() < epsilon:
                        action = int(rng.integers(num_actions))
                    else:
                        action = int(policy[state])
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode.append((int(state), action, float(reward)))
                    state = next_state
                    if terminated or truncated:
                        successes += int(reward > 0)
                        break
                for visited_state, visited_action, total in first_visit_returns(episode, gamma):
                    visit_counts[visited_state, visited_action] += 1
                    count = visit_counts[visited_state, visited_action]
                    q_values[visited_state, visited_action] += (
                        total - q_values[visited_state, visited_action]
                    ) / count

            for state in range(num_states):
                best = np.flatnonzero(q_values[state] == q_values[state].max())
                policy[state] = int(rng.choice(best))
            max_change = float(np.max(np.abs(q_values - previous)))
            metrics.append(
                {
                    "iteration": iteration + 1,
                    "epsilon": epsilon,
                    "training_success_rate": successes / episodes_per_iteration,
                    "max_q_change": max_change,
                }
            )
            epsilon = max(epsilon_min, epsilon - epsilon_decay)
            if (
                convergence_tolerance is not None
                and iteration + 1 >= min_iterations
                and max_change < convergence_tolerance
            ):
                converged = True
                break

        evaluation_success_rate = evaluate_policy(env, policy, 500, seed + 1)
    finally:
        env.close()
    return {
        "seed": seed,
        "slippery": slippery,
        "iterations_completed": len(metrics),
        "converged": converged,
        "premature_zero_convergence": bool(
            converged and metrics[-1]["training_success_rate"] == 0 and metrics[-1]["max_q_change"] == 0
        ),
        "evaluation_success_rate": evaluation_success_rate,
        "policy": policy.tolist(),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="First-visit Monte Carlo control on FrozenLake.")
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--episodes-per-iteration", type=positive_int, default=100)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-start", type=float, default=0.5)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.01)
    parser.add_argument("--convergence-tolerance", type=float)
    parser.add_argument("--min-iterations", type=positive_int, default=1)
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be in [0, 1]")
    if not 0 <= args.epsilon_min <= args.epsilon_start <= 1:
        parser.error("epsilon values must satisfy 0 <= min <= start <= 1")
    if args.epsilon_decay < 0:
        parser.error("--epsilon-decay must be non-negative")
    emit_json(
        monte_carlo_control(
            args.iterations,
            args.episodes_per_iteration,
            args.gamma,
            args.epsilon_start,
            args.epsilon_min,
            args.epsilon_decay,
            args.convergence_tolerance,
            args.min_iterations,
            args.seed,
            args.slippery,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()

