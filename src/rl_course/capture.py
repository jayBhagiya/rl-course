from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_course.blackjack import basic_policy
from rl_course.common import positive_int, write_json
from rl_course.neural_network import train
from rl_course.sysadmin import DEFAULT_INSTANCE, CrashedFirstPolicy, RandomPolicy, SysadminEnv
from rl_course.tic_tac_toe import (
    CIRCLE,
    CROSS,
    DEFAULT_POLICY_DIR,
    EMPTY,
    QLearningAgent,
    TicTacToeEnv,
    load_opponent,
    train_agent,
)


def capture_blackjack_policy() -> dict:
    rng = np.random.default_rng(0)
    states = []
    for usable_ace in (False, True):
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                action = basic_policy((player_sum, dealer_card, int(usable_ace)), rng)
                states.append(
                    {
                        "player_sum": player_sum,
                        "dealer_card": dealer_card,
                        "usable_ace": usable_ace,
                        "action": "hit" if action else "stick",
                    }
                )
    return {"policy": "basic", "states": states}


def capture_mountain_car(seed: int = 21, max_steps: int = 1_000) -> dict:
    env = gym.make("MountainCarContinuous-v0")
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)
    frames = [
        {"step": 0, "position": float(state[0]), "velocity": float(state[1]), "reward": 0.0}
    ]
    total_return = 0.0
    terminated = False
    try:
        for step in range(1, max_steps + 1):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            frames.append(
                {
                    "step": step,
                    "position": float(state[0]),
                    "velocity": float(state[1]),
                    "action": float(action[0]),
                    "reward": float(reward),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "environment": "MountainCarContinuous-v0",
        "policy": "random",
        "seed": seed,
        "success": bool(terminated),
        "total_return": total_return,
        "frames": frames,
    }


def capture_sysadmin(seed: int = 21, horizon: int = 40) -> dict:
    captures = []
    for policy_name, policy_type in (
        ("random", RandomPolicy),
        ("crashed-first", CrashedFirstPolicy),
    ):
        env = SysadminEnv()
        policy = policy_type(seed)
        state, _ = env.reset(seed=seed)
        frames = [{"step": 0, "state": state.tolist(), "fraction_crashed": 0.0}]
        total_return = 0.0
        try:
            for step in range(1, horizon + 1):
                action = policy(state)
                state, reward, _, _, _ = env.step(action)
                total_return += float(reward)
                frames.append(
                    {
                        "step": step,
                        "state": state.tolist(),
                        "maintained_server": action,
                        "reward": float(reward),
                        "fraction_crashed": env.fraction_crashed,
                    }
                )
        finally:
            env.close()
        captures.append(
            {"policy": policy_name, "seed": seed, "total_return": total_return, "frames": frames}
        )
    return {
        "num_servers": DEFAULT_INSTANCE.num_servers,
        "connections": [list(connection) for connection in DEFAULT_INSTANCE.connections],
        "horizon": horizon,
        "policies": captures,
    }


def capture_tic_tac_toe_game(
    agent: QLearningAgent,
    env: TicTacToeEnv,
    seed: int | None,
) -> dict:
    state, _ = env.reset(seed=seed)
    empty_board = np.full((3, 3), EMPTY, dtype=np.int8)
    frames = [{"actor": "reset", "board": empty_board.tolist()}]
    initial_moves = np.argwhere(state == CIRCLE)
    if len(initial_moves):
        move = initial_moves[0].tolist()
        frames.append({"actor": "opponent", "action": move, "board": state.tolist()})
    while True:
        action = agent.act(state, 0.0)
        row, column = divmod(action, 3)
        after_agent = state.copy()
        after_agent[row, column] = CROSS
        frames.append({"actor": "agent", "action": [row, column], "board": after_agent.tolist()})
        next_state, reward, terminated, _, _ = env.step((row, column))
        opponent_moves = np.argwhere((next_state == CIRCLE) & (after_agent != CIRCLE))
        if len(opponent_moves):
            move = opponent_moves[0].tolist()
            frames.append({"actor": "opponent", "action": move, "board": next_state.tolist()})
        state = next_state
        if terminated:
            outcome = {1: "win", 0: "draw", -1: "loss"}[int(reward)]
            return {"outcome": outcome, "frames": frames}


def capture_tic_tac_toe(seed: int = 21, training_episodes: int = 5_000) -> dict:
    captures = []
    for opponent_name in ("policy1", "policy2", "policy3", "policy4"):
        opponent = load_opponent(DEFAULT_POLICY_DIR / f"{opponent_name}.json")
        agent = train_agent(opponent, episodes=training_episodes, seed=seed)
        env = TicTacToeEnv(opponent)
        games = {}
        try:
            for episode in range(1_000):
                game = capture_tic_tac_toe_game(
                    agent,
                    env,
                    seed + 1 if episode == 0 else None,
                )
                games.setdefault(game["outcome"], game)
                if len(games) == 3:
                    break
        finally:
            env.close()
        captures.append(
            {
                "opponent": opponent_name,
                "seed": seed,
                "training_episodes": training_episodes,
                "states_learned": len(agent.q_values),
                "games": games,
            }
        )
    return {"marks": {"circle": CIRCLE, "empty": EMPTY, "cross": CROSS}, "policies": captures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate small browser-ready RL visual captures.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--tic-tac-toe-episodes", type=positive_int, default=5_000)
    args = parser.parse_args()

    neural_network = train(steps=2_000, batch_size=64, seed=args.seed, capture=True)
    neural_network["outputs"] = ["sum_of_squares", "sum"]
    outputs = {
        "blackjack-policy.json": capture_blackjack_policy(),
        "mountain-car.json": capture_mountain_car(args.seed),
        "neural-network.json": neural_network,
        "sysadmin.json": capture_sysadmin(args.seed),
        "tic-tac-toe.json": capture_tic_tac_toe(args.seed, args.tic_tac_toe_episodes),
    }
    for name, payload in outputs.items():
        write_json(payload, args.output_dir / name)
    print(json.dumps({"output_dir": str(args.output_dir), "files": sorted(outputs)}))


if __name__ == "__main__":
    main()
