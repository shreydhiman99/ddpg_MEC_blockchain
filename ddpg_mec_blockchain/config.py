import torch

# ─── System Parameters (from paper Table 1) ───────────────────────────────────
NUM_SERVERS = 20
NUM_TASKS = 50
DATA_SIZE_RANGE = (10, 20)
MAX_DATA_SIZE_SERVER = (200, 500)
UNIT_PRICE_RANGE = (1, 10)
TIME_CONSTRAINT_RANGE = (0.05, 0.3)
CPU_FREQ_RANGE = (1, 10)
TRANSMISSION_POWER_RANGE = (10, 20)
CHANNEL_GAIN_RANGE = (10, 20)
BANDWIDTH_RANGE = (10, 20)
GAUSSIAN_NOISE = 0.01
CPU_ARCH_PARAM = 0.01
CPU_CYCLES_PER_UNIT = 0.01

# ─── DDPG Hyperparameters ──────────────────────────────────────────────────────
DDPG_CONFIG = {
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_size": 100000,
    "noise_std": 0.2,
    "noise_clip": 0.5,
    "warmup_steps": 64,
    "updates_per_step": 10,
    "hidden_dims": [256, 256],
    "max_episodes": 500,
    "max_steps_per_episode": 200,
}

# ─── TD3 Hyperparameters ───────────────────────────────────────────────────────
TD3_CONFIG = {
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_size": 100000,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_delay": 2,
    "hidden_dims": [256, 256],
    "max_episodes": 500,
    "max_steps_per_episode": 200,
}

# ─── SAC Hyperparameters ───────────────────────────────────────────────────────
SAC_CONFIG = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_size": 100000,
    "hidden_dims": [256, 256],
    "max_episodes": 500,
    "max_steps_per_episode": 200,
}

# ─── Q-Learning Hyperparameters ────────────────────────────────────────────────
QLEARNING_CONFIG = {
    "alpha": 0.01,
    "gamma": 0.5,
    "epsilon": 0.5,
    "max_episodes": 500,
}

# ─── Experiment Parameters ─────────────────────────────────────────────────────
EXPERIMENT_CONFIG = {
    "num_runs": 5,
    "eval_interval": 10,
    "save_interval": 50,
    "server_range": [10, 20, 30, 40, 50, 60],
    "task_range": [50, 60, 70, 80, 90, 100],
    "unit_price_percentages": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "data_size_percentages":  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "task_size_range_mbps": list(range(59, 145, 5)),
    "learning_rate_values": [0.001, 0.005, 0.01, 0.05, 0.1],
    "discount_factor_values": [0.1, 0.3, 0.5, 0.7, 0.9],
    "random_seed": 42,
}

# ─── Blockchain Config ─────────────────────────────────────────────────────────
BLOCKCHAIN_CONFIG = {
    "ganache_url": "http://127.0.0.1:8545",
    "chain_id": 1337,
    "gas_limit": 3000000,
    "use_blockchain": False,
}

# ─── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
