"""
Main training script.
Usage: python experiments/train.py --agent ddpg --episodes 500
"""

import argparse
import numpy as np
import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from environment.mec_env import MECEnvironment
from environment.blockchain_logger import BlockchainLogger
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.qlearning_agent import QLearningAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from config import NUM_SERVERS, NUM_TASKS


_SOFT_REWARD_AGENTS = {"ddpg", "td3"}


def make_agent(agent_name, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if agent_name == "ddpg":
        return DDPGAgent(state_dim, action_dim, env.num_servers, env.num_tasks)
    elif agent_name == "td3":
        return TD3Agent(state_dim, action_dim, env.num_servers, env.num_tasks)
    elif agent_name == "qlearning":
        return QLearningAgent(state_dim, action_dim, env.num_servers, env.num_tasks)
    elif agent_name == "greedy":
        return GreedyAgent(env)
    elif agent_name == "random":
        return RandomAgent(action_dim)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def train(agent_name, num_episodes=500, num_servers=NUM_SERVERS, num_tasks=NUM_TASKS,
          use_blockchain=False, seed=42, save_dir="results/data", config_overrides=None):

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)

    env = MECEnvironment(num_servers=num_servers, num_tasks=num_tasks, seed=seed)

    # Apply config overrides (for ablation experiments)
    if config_overrides:
        import config as cfg_module
        for key, val in config_overrides.items():
            setattr(cfg_module, key, val)

    agent = make_agent(agent_name, env)
    blockchain = BlockchainLogger(use_blockchain=use_blockchain)

    episode_rewards = []
    episode_utilities = []
    episode_delays = []
    episode_energies = []
    episode_throughputs = []

    for episode in tqdm(range(num_episodes), desc=f"{agent_name}(s={num_servers},t={num_tasks},seed={seed})", leave=False):
        state = env.reset()
        action = agent.select_action(state)

        # DDPG/TD3: use differentiable soft reward for training; hard reward for reporting
        if agent_name in _SOFT_REWARD_AGENTS:
            train_reward = env.compute_soft_reward(action)
            next_state, _, done, info = env.step(action)
        else:
            next_state, train_reward, done, info = env.step(action)

        if use_blockchain:
            assignment = info["assignment"]
            for task_idx, server_idx in enumerate(assignment):
                blockchain.log_task_allocation(
                    task_id=episode * num_tasks + task_idx,
                    server_id=int(server_idx),
                    unit_price=float(env.tasks["unit_price"][task_idx]),
                    data_size=float(env.tasks["data_size"][task_idx]),
                    time_constraint=float(env.tasks["time_constraint"][task_idx]),
                    utility=float(info["task_utilities"][task_idx]),
                )
            blockchain.log_round(episode, info["total_utility"], info["completed_tasks"])

        agent.store(state, action, train_reward, next_state, done)
        agent.update()

        episode_rewards.append(float(info["total_utility"]))  # always log hard utility
        episode_utilities.append(float(info["total_utility"]))
        episode_delays.append(float(info["avg_delay"]))
        episode_energies.append(float(info["energy"]))
        episode_throughputs.append(float(info["throughput"]))

    tail = episode_rewards[-50:]
    results = {
        "agent": agent_name,
        "num_servers": num_servers,
        "num_tasks": num_tasks,
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_utilities": episode_utilities,
        "episode_delays": episode_delays,
        "episode_energies": episode_energies,
        "episode_throughputs": episode_throughputs,
        "mean_reward": float(np.mean(tail)),
        "std_reward": float(np.std(tail)),
        "mean_delay": float(np.mean(episode_delays[-50:])),
        "mean_energy": float(np.mean(episode_energies[-50:])),
        "mean_throughput": float(np.mean(episode_throughputs[-50:])),
    }

    fname = f"{save_dir}/{agent_name}_s{num_servers}_t{num_tasks}_seed{seed}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="ddpg", choices=["ddpg", "td3", "qlearning", "greedy", "random"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--servers", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--blockchain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train(args.agent, args.episodes, args.servers, args.tasks, args.blockchain, args.seed)
    print(f"Mean reward (last 50): {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
