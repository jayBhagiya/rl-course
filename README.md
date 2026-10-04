# Reinforcement Learning: From Fixed Rules to Deep Q-Networks

Course exercises and demos from *Reinforcement Learning* (Saarland University), rebuilt as reproducible Python experiments. They trace the path from evaluating fixed policies, through Monte Carlo control and tabular Q-learning, to a deep Q-network with experience replay and a target network. Course notebooks and exercise sheets are not part of this repository; the implementations are independent. The findings, including the failure cases, are in the [project write-up](https://jaybhagiya.me/projects/reinforcement-learning-coursework/).

## Experiments

| Module | What it does |
|---|---|
| `rl_course.blackjack` | Evaluates a random and a heuristic (`basic`) Blackjack policy |
| `rl_course.mountain_car` | Runs random actions in `MountainCarContinuous` |
| `rl_course.neural_network` | Fits a small PyTorch network to two analytic functions |
| `rl_course.sysadmin` | Evaluates a random and a `crashed-first` maintenance policy in the Sysadmin MDP |
| `rl_course.frozen_lake_mc` | First-visit Monte Carlo control on FrozenLake, with configurable convergence checks |
| `rl_course.tic_tac_toe` | Tabular Q-learning against four fixed opponent policies |
| `rl_course.racetrack` | Compares random, tabular Q-learning, online DQN, DQN with replay, and DQN with replay and a target network on Racetrack |

## Repository layout

| Path | Contents |
|---|---|
| `src/rl_course/` | One module per experiment, plus `capture.py` and `media.py` for the write-up's figures and videos |
| `data/opponent_policies/` | The four Tic-Tac-Toe opponent policies from the course |
| `tests/` | Smoke tests for every experiment |
| `condor/` | HTCondor jobs for the full campaigns |

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.10 or newer and the locked dependencies, including a CPU build of PyTorch:

```bash
git clone https://github.com/jayBhagiya/rl-course.git
cd rl-course
uv sync
```

## Running the experiments

Everything runs on a laptop CPU. Each command prints a JSON summary; add `--output PATH` to save it, and `--help` for all options. The commands below take between 2 and 11 seconds each:

```bash
uv run python -m rl_course.blackjack --policy basic --episodes 10000
uv run python -m rl_course.mountain_car --episodes 10
uv run python -m rl_course.neural_network --steps 2000
uv run python -m rl_course.sysadmin --episodes 100 --horizon 100
uv run python -m rl_course.frozen_lake_mc --iterations 50
uv run python -m rl_course.tic_tac_toe --opponent policy1 --episodes 5000
uv run python -m rl_course.racetrack --algorithm q-learning --episodes 1000
```

Racetrack writes `summary.json` and `metrics.csv` (evaluation return and success rate over training) into its output folder, by default `outputs/racetrack/<algorithm>-seed-<seed>/`. The full comparison trains for much longer than the example above: 300,000 episodes for tabular Q-learning and online DQN, 30,000 for DQN with replay, and 20,000 for DQN with a target network (see `condor/racetrack-full.sub` for every setting). That is why the campaigns run on a cluster.

## Trained agent

The DQN agent behind the write-up's Racetrack video (seed 87, 20,000 episodes, 98% evaluation success) is attached to the [v1.0 release](https://github.com/jayBhagiya/rl-course/releases/tag/v1.0). The archive holds `checkpoint.pt`, 100 evaluation `trajectories.json`, and the run's `metrics.csv` and `summary.json`:

```bash
mkdir -p runs
curl -LO https://github.com/jayBhagiya/rl-course/releases/download/v1.0/racetrack-dqn-seed-87.tar.gz
tar xzf racetrack-dqn-seed-87.tar.gz -C runs
```

```python
import torch
from rl_course.racetrack import build_agent

checkpoint = torch.load(
    "runs/rl-artifacts/racetrack/dqn/seed-87/checkpoint.pt", map_location="cpu", weights_only=True
)
agent = build_agent(
    "dqn", checkpoint["state_size"], checkpoint["num_actions"],
    alpha=5e-4, gamma=0.99, seed=checkpoint["seed"], device="cpu",
)
agent.online.load_state_dict(checkpoint["online_state_dict"])
action = agent.act(state)  # greedy action for a Racetrack observation
```

## Figures and videos for the write-up

`capture.py` writes browser-ready JSON for the small demos, and `media.py` replays saved policies and trajectories into MP4 clips with WebP posters (it needs `ffmpeg` on `PATH`). `media.py` reads two FrozenLake runs and the trained agent's trajectories, so create those first:

```bash
uv run python -m rl_course.frozen_lake_mc --iterations 50 --episodes-per-iteration 100 \
  --seed 21 --output runs/rl-v1/frozen-lake/full/seed-21.json
uv run python -m rl_course.frozen_lake_mc --iterations 50 --episodes-per-iteration 100 \
  --convergence-tolerance 0.5 --min-iterations 1 --seed 87 --output runs/rl-v1/frozen-lake/naive/seed-87.json
# plus the release archive extracted into runs/ (see above)

uv run python -m rl_course.capture --output-dir artifacts/local
uv run python -m rl_course.media --output-dir artifacts/media
```

## Running on an HTCondor cluster

`condor/` runs every campaign as CPU jobs inside the `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` Docker image, each with seeds 21, 42, and 87. A setup job installs uv and the locked environment once into shared storage; every other job reuses it.

| Submit file | Campaign | Jobs |
|---|---|---|
| `setup.sub` | Environment | 1 |
| `demos.sub` | Blackjack, Mountain Car, and the neural network | 12 |
| `coursework.sub` | Sysadmin, FrozenLake (three convergence settings), and Tic-Tac-Toe | 27 |
| `racetrack.sub` | Racetrack pilot: a random baseline and 10,000 episodes per learning method | 15 |
| `racetrack-full.sub` | Racetrack full comparison | 12 |
| `racetrack-artifacts.sub` | The DQN agent in the release, with `--save-artifacts` | 1 |

**1. Edit the variables at the top of each `.sub` file:**

| Variable | Set it to |
|---|---|
| `project_dir` | Where this repository is cloned, on a path the worker nodes can read |
| `data_dir` | Shared storage for the environment, caches, logs, and results (plan for about 5 GB) |
| `campaign` | Optional: the folder name for this set of runs under `data_dir/runs/` |

**2. Adapt the resource lines to your cluster:**
- The jobs request CPUs only: the networks are small and environment interaction is sequential. Add a `requirements` line if your cluster needs constraints such as a `UidDomain` or machine pool.
- `+WantGPUHomeMounted = true` is a site-specific attribute that mounts the home directory in the container. Remove it if your cluster doesn't define it.
- Your cluster must support the Docker universe. If it doesn't, switch to `universe = vanilla` and make sure the workers have Python available for `run.sh setup`.

**3. Create the log folder and submit:**

```bash
mkdir -p /path/to/large-storage/rl-course/logs

condor_submit -batch-name rl-setup condor/setup.sub    # once: environment
condor_submit -batch-name rl-demos condor/demos.sub
condor_submit -batch-name rl-coursework condor/coursework.sub
condor_submit -batch-name rl-racetrack condor/racetrack-full.sub
```

Results land in `data_dir/runs/<campaign>/` and job logs in `data_dir/logs/`. Keep campaign names distinct when rerunning, so earlier results aren't overwritten.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
