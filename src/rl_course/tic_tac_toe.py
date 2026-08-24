from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_course.common import emit_json, positive_int

CIRCLE, EMPTY, CROSS = 0, 1, 2
DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "opponent_policies"


def lines(board: np.ndarray) -> list[np.ndarray]:
    return [*board, *board.T, np.diag(board), np.diag(np.fliplr(board))]


def game_result(board: np.ndarray) -> int | None:
    for line in lines(board):
        if np.all(line == CROSS):
            return CROSS
        if np.all(line == CIRCLE):
            return CIRCLE
    return EMPTY if not np.any(board == EMPTY) else None


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent_policy: dict[str, list[list[int]]] | None = None) -> None:
        super().__init__()
        self.opponent_policy = opponent_policy
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.MultiDiscrete(np.full((3, 3), 3))
        self._state = np.full((3, 3), EMPTY, dtype=np.int8)

    def valid_actions(self) -> np.ndarray:
        return np.flatnonzero(self._state.reshape(-1) == EMPTY)

    def _perform(self, action: tuple[int, int], mark: int) -> None:
        row, column = action
        if self._state[row, column] != EMPTY:
            raise ValueError(f"field {action} is occupied")
        self._state[row, column] = mark

    def _opponent_action(self) -> tuple[int, int]:
        if self.opponent_policy is None:
            action = int(self.np_random.choice(self.valid_actions()))
            return divmod(action, 3)
        candidates = self.opponent_policy.get(str(self._state.tolist()))
        if not candidates:
            raise KeyError(f"opponent policy has no action for state {self._state.tolist()}")
        action = candidates[int(self.np_random.integers(len(candidates)))]
        return int(action[0]), int(action[1])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state.fill(EMPTY)
        if self.np_random.random() < 0.5:
            self._perform(self._opponent_action(), CIRCLE)
        return self._state.copy(), {}

    def step(self, action):
        action_array = np.asarray(action, dtype=int)
        if not self.action_space.contains(action_array):
            raise ValueError(f"invalid action {action}")
        selected = int(action_array[0]), int(action_array[1])
        self._perform(selected, CROSS)
        result = game_result(self._state)
        if result == CROSS:
            return self._state.copy(), 1.0, True, False, {}
        if result == EMPTY:
            return self._state.copy(), 0.0, True, False, {}
        self._perform(self._opponent_action(), CIRCLE)
        result = game_result(self._state)
        if result == CIRCLE:
            return self._state.copy(), -1.0, True, False, {}
        if result == EMPTY:
            return self._state.copy(), 0.0, True, False, {}
        return self._state.copy(), 0.0, False, False, {}

    def render(self) -> str:
        symbols = {CIRCLE: "O", EMPTY: " ", CROSS: "X"}
        return "\n".join("|".join(symbols[int(value)] for value in row) for row in self._state)


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q_values: dict[bytes, np.ndarray] = {}

    def _values(self, state: np.ndarray) -> np.ndarray:
        return self.q_values.setdefault(state.tobytes(), np.zeros(9, dtype=float))

    def act(self, state: np.ndarray, epsilon: float) -> int:
        valid = np.flatnonzero(state.reshape(-1) == EMPTY)
        if not len(valid):
            raise ValueError("terminal state has no valid actions")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        values = self._values(state)
        best = valid[values[valid] == values[valid].max()]
        return int(self.rng.choice(best))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        values = self._values(state)
        target = reward
        if not terminated:
            valid = np.flatnonzero(next_state.reshape(-1) == EMPTY)
            target += self.gamma * float(self._values(next_state)[valid].max())
        values[action] += self.alpha * (target - values[action])


def load_opponent(path: Path) -> dict[str, list[list[int]]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def train_agent(
    opponent_policy: dict[str, list[list[int]]],
    episodes: int = 5_000,
    alpha: float = 0.1,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    seed: int = 42,
) -> QLearningAgent:
    env = TicTacToeEnv(opponent_policy)
    agent = QLearningAgent(alpha=alpha, seed=seed)
    epsilon = epsilon_start
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            while True:
                action = agent.act(state, epsilon)
                next_state, reward, terminated, _, _ = env.step(divmod(action, 3))
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                if terminated:
                    break
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
    finally:
        env.close()
    return agent


def evaluate_agent(
    agent: QLearningAgent,
    opponent_policy: dict[str, list[list[int]]],
    episodes: int = 500,
    seed: int = 43,
) -> dict[str, float | int]:
    env = TicTacToeEnv(opponent_policy)
    outcomes = {-1: 0, 0: 0, 1: 0}
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            while True:
                action = agent.act(state, 0.0)
                state, reward, terminated, _, _ = env.step(divmod(action, 3))
                if terminated:
                    outcomes[int(reward)] += 1
                    break
    finally:
        env.close()
    return {
        "evaluation_episodes": episodes,
        "wins": outcomes[1],
        "draws": outcomes[0],
        "losses": outcomes[-1],
        "win_rate": outcomes[1] / episodes,
        "mean_return": (outcomes[1] - outcomes[-1]) / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning for Tic-Tac-Toe.")
    parser.add_argument("--opponent", choices=("policy1", "policy2", "policy3", "policy4"), default="policy1")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--episodes", type=positive_int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=positive_int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0 < args.epsilon_decay <= 1:
        parser.error("--epsilon-decay must be in (0, 1]")
    policy = load_opponent(args.policy_dir / f"{args.opponent}.json")
    agent = train_agent(
        policy,
        args.episodes,
        args.alpha,
        args.epsilon_start,
        args.epsilon_end,
        args.epsilon_decay,
        args.seed,
    )
    result = evaluate_agent(agent, policy, args.evaluation_episodes, args.seed + 1)
    result.update(
        {
            "opponent": args.opponent,
            "training_episodes": args.episodes,
            "states_learned": len(agent.q_values),
            "seed": args.seed,
        }
    )
    emit_json(result, args.output)


if __name__ == "__main__":
    main()
