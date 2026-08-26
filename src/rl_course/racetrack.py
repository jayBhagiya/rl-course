from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from torch import nn

from rl_course.common import emit_json, positive_int, write_json

_REGISTERED = False


def make_env(rt_args: str = "-sww -n -np 0.05") -> gym.Env:
    global _REGISTERED
    if not _REGISTERED:
        from racetrackgym import register_racetrack_envs

        register_racetrack_envs()
        _REGISTERED = True
    env = gym.make(
        "racetrack-barto_small-v0",
        max_episode_steps=100,
        rt_args=rt_args,
    )
    return FlattenObservation(env)


def state_key(state: np.ndarray) -> tuple[int, int, int, int]:
    values = np.asarray(state)[-4:]
    if len(values) != 4:
        raise ValueError("Racetrack observation must contain position and velocity")
    return tuple(int(round(float(value))) for value in values)


class RandomAgent:
    def __init__(self, num_actions: int, seed: int) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        return int(self.rng.integers(self.num_actions))

    def observe(self, *args) -> float | None:
        return None


class QLearningAgent:
    def __init__(
        self,
        num_actions: int,
        alpha: float,
        gamma: float,
        seed: int,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q_values: defaultdict[tuple[int, int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=float)
        )

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))
        values = self.q_values[state_key(state)]
        best = np.flatnonzero(values == values.max())
        return int(best[0] if epsilon == 0 else self.rng.choice(best))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> float | None:
        values = self.q_values[state_key(state)]
        target = reward
        if not terminated:
            target += self.gamma * float(self.q_values[state_key(next_state)].max())
        error = target - values[action]
        values[action] += self.alpha * error
        return abs(float(error))


class QNetwork(nn.Module):
    def __init__(self, state_size: int, num_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        num_actions: int,
        learning_rate: float,
        gamma: float,
        seed: int,
        device: str,
        replay: bool,
        target_network: bool,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        update_every: int = 10,
        tau: float = 0.001,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.online = QNetwork(state_size, num_actions).to(self.device)
        self.target = QNetwork(state_size, num_actions).to(self.device) if target_network else None
        if self.target is not None:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = replay
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.steps = 0
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))
        inputs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(inputs).argmax(dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> float | None:
        transition = (np.asarray(state), action, reward, np.asarray(next_state), terminated)
        if not self.replay:
            return self._learn([transition])
        self.buffer.append(transition)
        self.steps += 1
        if self.steps % self.update_every or len(self.buffer) < self.batch_size:
            return None
        indices = self.rng.choice(len(self.buffer), self.batch_size, replace=False)
        return self._learn([self.buffer[int(index)] for index in indices])

    def _learn(self, transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool]]) -> float:
        states = torch.as_tensor(
            np.stack([item[0] for item in transitions]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [item[1] for item in transitions], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            [item[2] for item in transitions], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item[3] for item in transitions]), dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            [item[4] for item in transitions], dtype=torch.float32, device=self.device
        )
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        bootstrap_network = self.target if self.target is not None else self.online
        with torch.no_grad():
            targets = rewards + self.gamma * bootstrap_network(next_states).max(dim=1).values * (1 - terminated)
        loss = nn.functional.mse_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.target is not None:
            with torch.no_grad():
                for target_parameter, online_parameter in zip(
                    self.target.parameters(), self.online.parameters()
                ):
                    target_parameter.mul_(1 - self.tau).add_(online_parameter, alpha=self.tau)
        return float(loss.item())


def evaluate(agent, env: gym.Env, episodes: int, seed: int) -> tuple[float, float]:
    returns: list[float] = []
    successes = 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total = 0.0
        last_reward = 0.0
        while True:
            action = agent.act(state, 0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            last_reward = float(reward)
            total += last_reward
            if terminated or truncated:
                successes += int(last_reward > 0)
                break
        returns.append(total)
    return float(np.mean(returns)), successes / episodes


def capture_episode(agent, env: gym.Env, seed: int | None) -> dict:
    state, _ = env.reset(seed=seed)
    x, y, velocity_x, velocity_y = state_key(state)
    frames = [
        {
            "step": 0,
            "position": [x, y],
            "velocity": [velocity_x, velocity_y],
            "reward": 0.0,
        }
    ]
    total_return = 0.0
    step = 0
    while True:
        step += 1
        action = agent.act(state, 0.0)
        state, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        x, y, velocity_x, velocity_y = state_key(state)
        frames.append(
            {
                "step": step,
                "position": [x, y],
                "velocity": [velocity_x, velocity_y],
                "action": action,
                "reward": float(reward),
                "experienced_noise": bool(info.get("experienced_noise", False)),
            }
        )
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            outcome = "success" if success else ("timeout" if truncated else "crash")
            return {"outcome": outcome, "total_return": total_return, "frames": frames}


def capture_evaluation_trajectories(
    agent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> dict:
    track = env.unwrapped.racetrack_env.map
    trajectories = {}
    for episode in range(episodes):
        trajectory = capture_episode(agent, env, seed if episode == 0 else None)
        trajectories.setdefault(trajectory["outcome"], trajectory)
        if "success" in trajectories and "crash" in trajectories:
            break
    return {
        "seed": seed,
        "episodes_searched": episode + 1,
        "track": {
            "height": track.height,
            "width": track.width,
            "rows": track.map,
            "starts": [list(position) for position in track.starters],
            "goals": [list(position) for position in track.goals],
        },
        "trajectories": trajectories,
    }


def save_dqn_artifacts(
    agent: DQNAgent,
    env: gym.Env,
    output: Path,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "algorithm": "dqn",
            "seed": seed,
            "state_size": agent.online.layers[0].in_features,
            "num_actions": agent.num_actions,
            "online_state_dict": agent.online.state_dict(),
            "target_state_dict": agent.target.state_dict() if agent.target is not None else None,
        },
        output / "checkpoint.pt",
    )
    write_json(
        capture_evaluation_trajectories(agent, env, episodes=100, seed=seed + 20_000),
        output / "trajectories.json",
    )


def build_agent(
    algorithm: str,
    state_size: int,
    num_actions: int,
    alpha: float,
    gamma: float,
    seed: int,
    device: str,
):
    if algorithm == "random":
        return RandomAgent(num_actions, seed)
    if algorithm == "q-learning":
        return QLearningAgent(num_actions, alpha, gamma, seed)
    return DQNAgent(
        state_size,
        num_actions,
        alpha,
        gamma,
        seed,
        device,
        replay=algorithm in {"replay-dqn", "dqn"},
        target_network=algorithm == "dqn",
    )


def train(
    algorithm: str,
    episodes: int,
    evaluation_interval: int,
    evaluation_episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    seed: int,
    device: str,
    rt_args: str,
    artifact_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    if artifact_dir is not None and algorithm != "dqn":
        raise ValueError("artifact saving currently supports dqn only")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_env = make_env(rt_args)
    evaluation_env = make_env(rt_args)
    state_size = int(np.prod(train_env.observation_space.shape))
    num_actions = int(train_env.action_space.n)
    agent = build_agent(algorithm, state_size, num_actions, alpha, gamma, seed, device)
    epsilon = epsilon_start
    interactions = 0
    recent_returns: deque[float] = deque(maxlen=100)
    metrics: list[dict] = []
    try:
        for episode in range(1, episodes + 1):
            state, _ = train_env.reset(seed=seed if episode == 1 else None)
            episode_return = 0.0
            losses: list[float] = []
            while True:
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = train_env.step(action)
                loss = agent.observe(state, action, float(reward), next_state, terminated)
                if loss is not None:
                    losses.append(loss)
                state = next_state
                episode_return += float(reward)
                interactions += 1
                if terminated or truncated:
                    break
            recent_returns.append(episode_return)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            if episode % evaluation_interval == 0 or episode == episodes:
                mean_return, success_rate = evaluate(
                    agent, evaluation_env, evaluation_episodes, seed + 10_000
                )
                metrics.append(
                    {
                        "episode": episode,
                        "environment_steps": interactions,
                        "epsilon": epsilon,
                        "training_return": episode_return,
                        "rolling_return": float(np.mean(recent_returns)),
                        "mean_loss": float(np.mean(losses)) if losses else None,
                        "evaluation_return": mean_return,
                        "evaluation_success_rate": success_rate,
                    }
                )
        if artifact_dir is not None:
            save_dqn_artifacts(agent, evaluation_env, artifact_dir, seed)
    finally:
        train_env.close()
        evaluation_env.close()
    summary = {
        "algorithm": algorithm,
        "episodes": episodes,
        "evaluation_episodes": evaluation_episodes,
        "environment_steps": interactions,
        "seed": seed,
        "device": device,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "rt_args": rt_args,
        "final_evaluation_return": metrics[-1]["evaluation_return"],
        "final_evaluation_success_rate": metrics[-1]["evaluation_success_rate"],
    }
    if artifact_dir is not None:
        summary["artifacts"] = ["checkpoint.pt", "trajectories.json"]
    return summary, metrics


def write_metrics(metrics: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tabular Q-learning and DQN variants.")
    parser.add_argument(
        "--algorithm",
        choices=("random", "q-learning", "online-dqn", "replay-dqn", "dqn"),
        default="q-learning",
    )
    parser.add_argument("--episodes", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-interval", type=positive_int, default=100)
    parser.add_argument("--evaluation-episodes", type=positive_int, default=20)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9996)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--rt-args", default="-sww -n -np 0.05")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-artifacts", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gamma <= 1:
        parser.error("--gamma must be in (0, 1]")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0 < args.epsilon_decay <= 1:
        parser.error("--epsilon-decay must be in (0, 1]")
    alpha = args.alpha if args.alpha is not None else (0.05 if args.algorithm == "q-learning" else 0.0005)
    if alpha <= 0:
        parser.error("--alpha must be positive")
    if args.save_artifacts and args.algorithm != "dqn":
        parser.error("--save-artifacts currently supports --algorithm dqn only")
    output = args.output or Path("outputs/racetrack") / f"{args.algorithm}-seed-{args.seed}"
    summary, metrics = train(
        args.algorithm,
        args.episodes,
        args.evaluation_interval,
        args.evaluation_episodes,
        alpha,
        args.gamma,
        args.epsilon_start,
        args.epsilon_end,
        args.epsilon_decay,
        args.seed,
        args.device,
        args.rt_args,
        output if args.save_artifacts else None,
    )
    write_metrics(metrics, output / "metrics.csv")
    emit_json(summary, output / "summary.json")


if __name__ == "__main__":
    main()
