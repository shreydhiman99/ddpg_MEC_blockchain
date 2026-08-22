"""
Run ALL experiments required for paper figures 1–7.
Usage: python experiments/run_all_baselines.py
Estimated runtime: 30–90 min on CPU.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.train import train
import config as cfg_module
from config import EXPERIMENT_CONFIG, NUM_SERVERS, NUM_TASKS

AGENTS = ["ddpg", "td3", "qlearning", "greedy", "random"]
NUM_RUNS = 3  # reduced for faster runs (was 5)
SAVE_DIR = "results/data"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_with_seeds(agent, num_servers, num_tasks, num_episodes=200):
    all_results = []
    for run in range(NUM_RUNS):
        seed = EXPERIMENT_CONFIG["random_seed"] + run
        result = train(agent, num_episodes, num_servers, num_tasks, seed=seed)
        all_results.append(result)
    return all_results


def experiment_vary_servers():
    print("\n=== Fig 1: Vary Number of Servers ===")
    for n_servers in EXPERIMENT_CONFIG["server_range"]:
        for agent in AGENTS:
            run_with_seeds(agent, n_servers, NUM_TASKS)


def experiment_vary_tasks():
    print("\n=== Fig 2: Vary Number of Tasks ===")
    for n_tasks in EXPERIMENT_CONFIG["task_range"]:
        for agent in AGENTS:
            run_with_seeds(agent, NUM_SERVERS, n_tasks)


def experiment_vary_unit_price():
    print("\n=== Fig 3: Vary Unit Price ===")
    orig_min, orig_max = cfg_module.UNIT_PRICE_RANGE
    for pct in EXPERIMENT_CONFIG["unit_price_percentages"]:
        cfg_module.UNIT_PRICE_RANGE = (orig_min * (1 + pct), orig_max * (1 + pct))
        for agent in ["ddpg", "qlearning"]:
            result = train(agent, 200, NUM_SERVERS, NUM_TASKS, seed=42)
            fname = f"{SAVE_DIR}/{agent}_unit_price_{pct:.1f}.json"
            with open(fname, "w") as f:
                json.dump({"pct": pct, "mean_reward": result["mean_reward"]}, f)
    cfg_module.UNIT_PRICE_RANGE = (orig_min, orig_max)


def experiment_vary_data_size():
    print("\n=== Fig 4: Vary Data Size ===")
    orig = cfg_module.DATA_SIZE_RANGE
    for pct in EXPERIMENT_CONFIG["data_size_percentages"]:
        cfg_module.DATA_SIZE_RANGE = (orig[0] * (1 + pct), orig[1] * (1 + pct))
        for agent in ["ddpg", "qlearning"]:
            result = train(agent, 200, NUM_SERVERS, NUM_TASKS, seed=42)
            fname = f"{SAVE_DIR}/{agent}_data_size_{pct:.1f}.json"
            with open(fname, "w") as f:
                json.dump({"pct": pct, "mean_reward": result["mean_reward"]}, f)
    cfg_module.DATA_SIZE_RANGE = orig


def experiment_end_to_end_delay():
    print("\n=== Fig 5: End-to-End Delay ===")
    orig = cfg_module.DATA_SIZE_RANGE
    for task_mbps in EXPERIMENT_CONFIG["task_size_range_mbps"]:
        cfg_module.DATA_SIZE_RANGE = (task_mbps * 0.9, task_mbps * 1.1)
        for agent in ["ddpg", "qlearning"]:
            result = train(agent, 200, NUM_SERVERS, NUM_TASKS, seed=42)
            fname = f"{SAVE_DIR}/{agent}_delay_tasksize_{task_mbps}.json"
            with open(fname, "w") as f:
                json.dump({"task_mbps": task_mbps, "mean_delay": result["mean_delay"]}, f)
    cfg_module.DATA_SIZE_RANGE = orig


def experiment_learning_rate():
    print("\n=== Fig 6: Learning Rate Ablation ===")
    orig_ddpg_lr = cfg_module.DDPG_CONFIG["actor_lr"]
    orig_ql_alpha = cfg_module.QLEARNING_CONFIG["alpha"]
    for alpha in EXPERIMENT_CONFIG["learning_rate_values"]:
        cfg_module.DDPG_CONFIG["actor_lr"] = alpha
        cfg_module.QLEARNING_CONFIG["alpha"] = alpha
        for agent in ["ddpg", "qlearning"]:
            result = train(agent, 200, NUM_SERVERS, NUM_TASKS, seed=42)
            fname = f"{SAVE_DIR}/{agent}_lr_{alpha}.json"
            with open(fname, "w") as f:
                json.dump({"alpha": alpha, "rewards": result["episode_rewards"]}, f)
    cfg_module.DDPG_CONFIG["actor_lr"] = orig_ddpg_lr
    cfg_module.QLEARNING_CONFIG["alpha"] = orig_ql_alpha


def experiment_discount_factor():
    print("\n=== Fig 7: Discount Factor Ablation ===")
    orig_ddpg_gamma = cfg_module.DDPG_CONFIG["gamma"]
    orig_ql_gamma = cfg_module.QLEARNING_CONFIG["gamma"]
    for gamma in EXPERIMENT_CONFIG["discount_factor_values"]:
        cfg_module.DDPG_CONFIG["gamma"] = gamma
        cfg_module.QLEARNING_CONFIG["gamma"] = gamma
        for agent in ["ddpg", "qlearning"]:
            result = train(agent, 200, NUM_SERVERS, NUM_TASKS, seed=42)
            fname = f"{SAVE_DIR}/{agent}_gamma_{gamma}.json"
            with open(fname, "w") as f:
                json.dump({"gamma": gamma, "rewards": result["episode_rewards"]}, f)
    cfg_module.DDPG_CONFIG["gamma"] = orig_ddpg_gamma
    cfg_module.QLEARNING_CONFIG["gamma"] = orig_ql_gamma


if __name__ == "__main__":
    print("Starting full experimental pipeline...")
    experiment_vary_servers()
    experiment_vary_tasks()
    experiment_vary_unit_price()
    experiment_vary_data_size()
    experiment_end_to_end_delay()
    experiment_learning_rate()
    experiment_discount_factor()
    print("\nAll experiments complete. Run: python analysis/plot_results.py")
