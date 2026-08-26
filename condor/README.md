# HTCondor experiments

These files target an HTCondor cluster. They use three seeds: 21, 42, and 87. All jobs currently request CPUs only. The Racetrack networks are small and environment interaction is sequential, so GPU transfer overhead is unlikely to help before profiling shows otherwise.

## Configure paths

Before submitting, update these values in each `.sub` file:

- `project_dir`: absolute path to this repository checkout
- `data_dir`: shared writable directory for environments, logs, and run outputs

## Create environment

Run on `your submit machine`:

```console
$ mkdir -p <data_dir>/logs
$ cd <project_dir>
$ condor_submit -batch-name rl-setup condor/setup.sub
$ condor_q
```

Wait for setup to finish successfully. It installs pinned `uv`, Python 3.10.16, and dependencies from `uv.lock` into `<data_dir>/venvs/rl-course-cpu`.

## Submit campaigns

```console
$ condor_submit -batch-name rl-demos condor/demos.sub
$ condor_submit -batch-name rl-coursework condor/coursework.sub
$ condor_submit -batch-name rl-racetrack-pilot condor/racetrack.sub
```

Campaign sizes:

- `demos.sub`: 12 jobs
- `coursework.sub`: 27 jobs
- `racetrack.sub`: 15 pilot jobs

Monitor jobs:

```console
$ condor_q
$ condor_q -hold
```

Results are written below `<data_dir>/runs/`. Condor output, error, and event logs are written below `<data_dir>/logs/`.

To capture one trained DQN checkpoint and representative Racetrack trajectories for the website:

```console
$ condor_submit -batch-name rl-racetrack-artifacts condor/racetrack-artifacts.sub
```

## Racetrack pilot

Pilot trains each learning method for 10,000 episodes and evaluates every 500 episodes. Review runtime and learning curves before increasing this budget. Keep campaign name distinct when rerunning, so earlier results are not overwritten.

## Weights & Biases

W&B is intentionally not used. JSON, CSV, and Condor logs already capture every metric needed for this project without adding network, authentication, or dependency failure modes. Add W&B only if Racetrack runs become long enough that remote live monitoring materially improves operation.
