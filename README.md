# Blockchain-MEC DDPG

Reinforcement-learning framework for task offloading / resource allocation in a
blockchain-based Mobile Edge Computing (MEC) system. A DDPG agent with a
cross-attention actor learns to assign tasks to edge servers to maximize
system utility, with optional on-chain logging of allocation decisions via a
Solidity smart contract. Implements the experiments behind the paper
*"Optimizing Resource Sharing in Blockchain-based Mobile Edge Computing
through Reinforcement Learning: A DDPG Approach"* (submitted to MTAP,
Springer (under review)).

## Architecture

```
                     ┌─────────────────────────┐
   state (tasks +    │   MECEnvironment         │   reward = system utility
   server features)  │   environment/mec_env.py │   (Eq. 7-10 of the paper)
       ┌──────────── │                          │ ◄───────────┐
       │             └─────────────────────────┘              │
       ▼                                                      │
┌─────────────────┐   action (S×T logits)   ┌─────────────────┴──────┐
│  Agent           │ ───────────────────────►│  step(): compute      │
│  (ddpg/td3/...)  │                         │  utility, feasi-      │
└─────────────────┘                          │  bility, next state   │
       │                                     └───────────────────────┘
       │ optional
       ▼
┌─────────────────────────┐        ┌─────────────────────────────────┐
│ BlockchainLogger         │───────►│ TaskAllocation.sol (Ganache)   │
│ environment/blockchain_  │        │ blockchain/contracts/          │
│ logger.py                │        │ on-chain allocation + round log│
└─────────────────────────┘        └─────────────────────────────────┘
```

### Environment (`environment/`)

- **`mec_env.py`** — `MECEnvironment`, a `gymnasium.Env`. State is a flattened
  `(num_tasks*5 + num_servers*5)` vector (task features: unit price, data
  size, time constraint, required CPU cycles, submitting server; server
  features: CPU frequency, bandwidth, tx power, channel gain, max CPU
  cycles). Action is an `(num_servers * num_tasks)` logit matrix. `step()`
  takes the `argmax` assignment per task, computes per-task utility (revenue
  minus computation/communication energy, zeroed out if infeasible on CPU
  capacity or time constraint), and returns the total utility as the reward
  plus an `info` dict with throughput, average delay, and energy.
  `compute_soft_reward()` provides a **differentiable** proxy reward
  (per-task softmax over the utility matrix) used to train the DDPG/TD3
  actors.
- **`task_generator.py`** — standalone helpers to sample server/task
  parameter sets (used outside the `Env` class, e.g. for the greedy baseline
  or scripts).
- **`blockchain_logger.py`** — `BlockchainLogger`, wraps a `web3.py` contract
  call per task/round; silently falls back to an in-memory local log if
  `use_blockchain=False` or Ganache is unreachable.

### Networks (`networks/`)

- **`attention_actor.py`** — `AttentionActor` (used by DDPG/TD3 by default).
  Tasks are embedded and used as **queries**, servers as **keys/values** in a
  multi-head cross-attention layer; a small head scores each (task, server)
  pair. This is the actor that made learning actually work — a monolithic
  MLP actor with ~1000-dim output has too few parameters/gradient signal to
  learn from ~500 episodes (see `Actor` below for the naive baseline). ~20K
  parameters vs. ~400K for the monolithic version.
- **`actor.py`** — `Actor`, a plain MLP (`state → tanh → action`). Used only
  when `num_servers`/`num_tasks` aren't passed to the agent (fallback).
- **`scoring_actor.py`** — `ScoringActor`, an alternative parameter-shared
  scorer (no attention, just an MLP over concatenated task/server feature
  pairs). Not used by default but kept as an ablation option.
- **`critic.py`** — `Critic` (state+action → Q-value) and `TwinCritic` (two
  independent critics, for TD3's clipped double-Q).
- **`replay_buffer.py`** — fixed-size numpy replay buffer.

### Agents (`agents/`)

| Agent | File | Notes |
|---|---|---|
| DDPG (ours) | `ddpg_agent.py` | `AttentionActor` + `Critic`, OU exploration noise, stores **raw logits** in the buffer and applies per-task softmax at update time |
| TD3 | `td3_agent.py` | `AttentionActor` + `TwinCritic`, delayed policy updates, target policy smoothing |
| Q-Learning | `qlearning_agent.py` | Discretized state/action tabular baseline |
| Greedy | `greedy_agent.py` | Assigns each task to the server maximizing `unit_price × cpu_cycles` |
| Random | `random_agent.py` | Uniform random logits |

All agents expose the same minimal interface: `select_action(state, evaluate=False)`,
`store(...)`, `update()`.

### Experiments & analysis

- **`experiments/train.py`** — trains a single agent for N episodes on a
  given `(num_servers, num_tasks, seed)` and dumps per-episode metrics to
  `results/data/<agent>_s<S>_t<T>_seed<seed>.json`. DDPG/TD3 train against
  the differentiable soft reward but log the hard (argmax) utility for
  reporting.
- **`experiments/run_all_baselines.py`** — orchestrates every experiment
  needed for the paper's figures (varying server count, task count, unit
  price, data size, task size/delay, learning rate, discount factor) across
  all 5 agents and 3 seeds. ~30–90 min on CPU.
- **`analysis/plot_results.py`** — reads `results/data/*.json` and renders
  `results/figures/fig1..fig7*.{png,pdf}` with confidence intervals.
- **`generate_results_table.py`** — aggregates the `s20_t50` runs into a
  LaTeX results table (`results/figures/results_table.tex`).
- **`generate_architecture.py`** — draws the system architecture diagram
  used in the paper.

### Blockchain (`blockchain/`)

- **`contracts/TaskAllocation.sol`** — Solidity 0.8 contract with
  `logAllocation` (per-task) and `logRound` (per-episode) admin-only
  functions, storing allocations/rounds and emitting events. Modeled as
  running on a PBFT consortium chain.
- **`deploy.py`** — compiles and deploys the contract to a local Ganache
  node, writing `blockchain/deployment.json` (address + ABI) for the logger
  to pick up.

### Config (`config.py`)

Single source of truth for system parameters (Table 1 of the paper),
per-algorithm hyperparameters (`DDPG_CONFIG`, `TD3_CONFIG`, `SAC_CONFIG`,
`QLEARNING_CONFIG`), experiment sweep ranges (`EXPERIMENT_CONFIG`), and
blockchain connection settings (`BLOCKCHAIN_CONFIG`). Several scripts mutate
`config` module attributes at runtime to run ablations (e.g. varying
learning rate) — `mec_env.py` re-reads `config` values on every task/server
generation so those overrides propagate.

## Setup

Requires Python 3.9+.

```bash
cd ddpg_mec_blockchain
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Blockchain logging is optional and only needed if you set
`use_blockchain=True`. It additionally requires:
- [Ganache](https://trufflesuite.com/ganache/) (CLI or GUI) for a local
  Ethereum testnet
- A working Solidity compiler, auto-installed on first use via `py-solc-x`
  (`install_solc("0.8.0")`)

All commands below assume your working directory is `ddpg_mec_blockchain/`
(the package root — `analysis`, `agents`, `environment`, etc. are imported as
top-level packages).

## Running

### Train a single agent

```bash
python experiments/train.py --agent ddpg --episodes 500 --servers 20 --tasks 50 --seed 42
```

Options: `--agent {ddpg,td3,qlearning,greedy,random}`, `--episodes`,
`--servers`, `--tasks`, `--seed`, `--blockchain` (enable on-chain logging).
Results are written to `results/data/<agent>_s<S>_t<T>_seed<seed>.json`.

### Reproduce all paper experiments

```bash
python experiments/run_all_baselines.py
```

Runs every sweep (server/task count, unit price, data size, task size/delay,
learning rate, discount factor) for all 5 agents × 3 seeds. Takes ~30–90
minutes on CPU. Output goes to `results/data/`.

### Generate figures and the results table

```bash
python -m analysis.plot_results
python generate_results_table.py
```

Figures land in `results/figures/*.{png,pdf}`; the LaTeX table in
`results/figures/results_table.tex`.

### Enable blockchain logging

```bash
ganache --port 8545                 # in a separate terminal
python blockchain/deploy.py         # deploys TaskAllocation.sol, writes deployment.json
python experiments/train.py --agent ddpg --blockchain
```

Or set `BLOCKCHAIN_CONFIG["use_blockchain"] = True` in `config.py` to make it
the default for all runs. If Ganache isn't reachable, `BlockchainLogger`
prints a warning and transparently falls back to local (in-memory) logging —
training still works without it.

## Key results (from `results/data/`)

DDPG (with the attention actor) reaches mean reward ≈ 19.3 ± 0.6 with
throughput 1.0 and average delay ≈ 0.041s on the default 20-server/50-task
setting, outperforming Q-Learning, Greedy, and Random baselines and matching
TD3. See `results/figures/`.
