# Reinforcement Learning Course Demos

Clean Python implementations of demos and programming exercises from Saarland University's Reinforcement Learning course. Original notebooks and exercise sheets remain source material; this repository contains independent, reproducible implementations.

## Setup

```console
$ uv sync
```

## Demos

```console
$ uv run python -m rl_course.blackjack --policy basic --episodes 10000
$ uv run python -m rl_course.mountain_car --episodes 10
$ uv run python -m rl_course.neural_network --steps 2000
```

## Exercises

```console
$ uv run python -m rl_course.sysadmin --episodes 100 --horizon 100
$ uv run python -m rl_course.frozen_lake_mc --iterations 50
$ uv run python -m rl_course.tic_tac_toe --opponent policy1 --episodes 5000
```

## Racetrack comparison

```console
$ uv run python -m rl_course.racetrack --algorithm q-learning --episodes 1000
$ uv run python -m rl_course.racetrack --algorithm dqn --episodes 5000
```

Commands print JSON summaries. Pass `--output PATH` to save results. Racetrack writes both `summary.json` and `metrics.csv` into its output directory.

## Checks

```console
$ uv run python -m unittest discover -s tests -v
```

Small demos and current Racetrack networks are CPU-friendly.

## HTCondor

Cluster setup, campaign submission, and result locations are documented in [condor/README.md](condor/README.md). Campaigns use three seeds: 21, 42, and 87.

