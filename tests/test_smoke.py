import unittest

import numpy as np
import torch

from rl_course.blackjack import basic_policy, evaluate
from rl_course.frozen_lake_mc import first_visit_returns, monte_carlo_control
from rl_course.neural_network import train
from rl_course.racetrack import DQNAgent, state_key
from rl_course.sysadmin import CRASHED, DEFAULT_INSTANCE, SysadminEnv, compute_crash_probabilities
from rl_course.tic_tac_toe import CROSS, EMPTY, TicTacToeEnv, game_result


class SmokeTests(unittest.TestCase):
    def test_blackjack_policy_and_run(self):
        rng = np.random.default_rng(1)
        self.assertEqual(basic_policy((20, 10, 0), rng), 0)
        self.assertEqual(basic_policy((11, 6, 0), rng), 1)
        self.assertEqual(evaluate(basic_policy, episodes=10, seed=1)["episodes"], 10)

    def test_first_visit_returns_include_repeated_rewards(self):
        episode = [(0, 0, 1.0), (1, 1, 2.0), (0, 0, 3.0)]
        self.assertEqual(first_visit_returns(episode, 1.0), [(0, 0, 6.0), (1, 1, 5.0)])

    def test_frozen_lake_smoke(self):
        result = monte_carlo_control(iterations=2, episodes_per_iteration=5, seed=1)
        self.assertEqual(result["iterations_completed"], 2)

    def test_neural_network_learns(self):
        result = train(steps=200, batch_size=32, seed=1)
        self.assertLess(result["final_mse"], result["initial_mse"])

    def test_sysadmin_probabilities_and_seed(self):
        state = np.array([CRASHED, 0, 0, 0, 0, 0], dtype=np.int8)
        probabilities = compute_crash_probabilities(state, 0, DEFAULT_INSTANCE)
        self.assertEqual(probabilities[0], 0)
        left, right = SysadminEnv(), SysadminEnv()
        left.reset(seed=1)
        right.reset(seed=1)
        self.assertTrue(np.array_equal(left.step(0)[0], right.step(0)[0]))

    def test_tic_tac_toe_random_opponent(self):
        board = np.full((3, 3), EMPTY, dtype=np.int8)
        board[0] = CROSS
        self.assertEqual(game_result(board), CROSS)
        env = TicTacToeEnv()
        state, _ = env.reset(seed=1)
        while True:
            action = int(np.flatnonzero(state.reshape(-1) == EMPTY)[0])
            state, _, terminated, _, _ = env.step(divmod(action, 3))
            if terminated:
                break
        env.close()

    def test_racetrack_agents_without_environment(self):
        self.assertEqual(state_key(np.arange(15)), (11, 12, 13, 14))
        agent = DQNAgent(
            state_size=4,
            num_actions=2,
            learning_rate=0.001,
            gamma=0.99,
            seed=1,
            device="cpu",
            replay=False,
            target_network=True,
        )
        for online, target in zip(agent.online.parameters(), agent.target.parameters()):
            self.assertTrue(torch.equal(online, target))
        loss = agent.observe(np.zeros(4), 0, 1.0, np.ones(4), True)
        self.assertGreaterEqual(loss, 0)


if __name__ == "__main__":
    unittest.main()
