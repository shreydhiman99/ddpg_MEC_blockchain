"""
MEC Environment for task allocation simulation.
State space: λ = <ϑ, B, M, ϑe>  (tasks x 4 matrix)
Action space: A = <C, Z, Q, L, O, Y, α, σ>  (servers x 8 matrix)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# Import as module so ablation experiments that rebind config names are visible here
import config as _cfg


class MECEnvironment(gym.Env):
    """
    Blockchain-based MEC environment.
    State = current set of tasks (costs, data sizes, time constraints)
    Action = allocation decisions (which server handles which task)
    Reward = total utility across all server-task pairs (Eq. 7-10)
    """

    def __init__(self, num_servers=None, num_tasks=None, seed=42):
        super().__init__()
        self.num_servers = num_servers if num_servers is not None else _cfg.NUM_SERVERS
        self.num_tasks = num_tasks if num_tasks is not None else _cfg.NUM_TASKS
        self.rng = np.random.default_rng(seed)

        # State = task features (5 per task) + server features (5 per server)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.num_tasks * 5 + self.num_servers * 5,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.num_servers * self.num_tasks,),
            dtype=np.float32
        )

        self.servers = None
        self.tasks = None
        self._generate_servers()

    def _generate_servers(self):
        # Access config values at call-time so ablation experiments propagate
        self.servers = {
            "max_cpu_cycles": self.rng.uniform(*_cfg.MAX_DATA_SIZE_SERVER, size=self.num_servers),
            "cpu_freq": self.rng.uniform(*_cfg.CPU_FREQ_RANGE, size=self.num_servers),
            "bandwidth": self.rng.uniform(*_cfg.BANDWIDTH_RANGE, size=self.num_servers),
            "tx_power": self.rng.uniform(*_cfg.TRANSMISSION_POWER_RANGE, size=self.num_servers),
            "channel_gain": self.rng.uniform(*_cfg.CHANNEL_GAIN_RANGE, size=self.num_servers),
            "cpu_arch_param": np.full(self.num_servers, _cfg.CPU_ARCH_PARAM),
            "cpu_cycles_per_unit": np.full(self.num_servers, _cfg.CPU_CYCLES_PER_UNIT),
            "unit_price": self.rng.uniform(*_cfg.UNIT_PRICE_RANGE, size=self.num_servers),
        }

    def _generate_tasks(self):
        # Access config values at call-time so ablation experiments propagate
        self.tasks = {
            "unit_price": self.rng.uniform(*_cfg.UNIT_PRICE_RANGE, size=self.num_tasks),
            "data_size": self.rng.uniform(*_cfg.DATA_SIZE_RANGE, size=self.num_tasks),
            "time_constraint": self.rng.uniform(*_cfg.TIME_CONSTRAINT_RANGE, size=self.num_tasks),
            "submitting_server": self.rng.integers(0, self.num_servers, size=self.num_tasks),
        }
        self.tasks["cpu_cycles_needed"] = (
            self.tasks["data_size"].reshape(1, -1) *
            self.servers["cpu_cycles_per_unit"].reshape(-1, 1)
        )

    def _get_state(self):
        # Task features (5 per task): unit_price, data_size, time_constraint, cpu_cycles, submitting_server
        submitting_normalized = self.tasks["submitting_server"].astype(np.float32) / self.num_servers
        task_block = np.stack([
            self.tasks["unit_price"],
            self.tasks["data_size"],
            self.tasks["time_constraint"],
            self.tasks["cpu_cycles_needed"][0],
            submitting_normalized,
        ], axis=1).flatten().astype(np.float32)

        # Server features (5 per server): all resource dimensions needed by scoring actor
        server_block = np.stack([
            self.servers["cpu_freq"],
            self.servers["bandwidth"],
            self.servers["tx_power"],
            self.servers["channel_gain"],
            self.servers["max_cpu_cycles"],
        ], axis=1).flatten().astype(np.float32)

        raw = np.concatenate([task_block, server_block])
        lo, hi = raw.min(), raw.max()
        return ((raw - lo) / (hi - lo + 1e-8)).astype(np.float32)

    def _compute_data_rate(self, server_idx):
        Lj = self.servers["bandwidth"][server_idx]
        Oj = self.servers["tx_power"][server_idx]
        Yj = self.servers["channel_gain"][server_idx]
        return Lj * np.log2(1 + (Oj * Yj) / (_cfg.GAUSSIAN_NOISE ** 2))

    def _compute_comm_time(self, server_idx, task_idx):
        nj = self._compute_data_rate(server_idx)
        return self.tasks["data_size"][task_idx] / nj

    def _compute_comm_energy(self, server_idx, task_idx):
        Oj = self.servers["tx_power"][server_idx]
        kcomm = self._compute_comm_time(server_idx, task_idx)
        return Oj * kcomm

    def _compute_comp_energy(self, server_idx, task_idx):
        alpha_j = self.servers["cpu_arch_param"][server_idx]
        phi_ji = self.tasks["cpu_cycles_needed"][server_idx, task_idx]
        qj = self.servers["cpu_freq"][server_idx]
        return alpha_j * phi_ji * (qj ** 2)

    def _compute_comp_time(self, server_idx, task_idx):
        phi_ji = self.tasks["cpu_cycles_needed"][server_idx, task_idx]
        qj = self.servers["cpu_freq"][server_idx]
        return phi_ji / qj

    def _compute_utility(self, server_idx, task_idx, assigned_server_idx):
        """
        System-level utility for assigning task_idx to assigned_server_idx.

        For a LOCAL task (submitter == assigned_server_idx):
          - Revenue: bi * phi (full, task stays local)
          - Cost: u_comp (computation energy only)
          - Time: comp_time must fit within time_constraint

        For an OFFLOADED task (submitter != assigned_server_idx):
          - Revenue: bi * phi (full, task executed elsewhere)
          - Cost: u_comp (at assigned server) + u_comm (comm from submitting server)
          - Time: comp_time + comm_time must fit within time_constraint
          - Offloading is worthwhile only if the assigned server is faster
            (lower comp_time) than the submitter despite the added comm cost.

        Infeasible assignments (CPU overflow or time violation): utility = 0.
        """
        submitter = self.tasks["submitting_server"][task_idx]
        bi = self.tasks["unit_price"][task_idx]
        phi_ji = self.tasks["cpu_cycles_needed"][server_idx, task_idx]
        phi_j = self.servers["max_cpu_cycles"][server_idx]
        time_constraint = self.tasks["time_constraint"][task_idx]

        comp_time = self._compute_comp_time(server_idx, task_idx)
        u_comp = self._compute_comp_energy(server_idx, task_idx)

        if submitter == server_idx:
            # Local: only computation, no communication
            total_time = comp_time
            u_total_cost = u_comp
        else:
            # Offloaded: add communication overhead from submitting server
            comm_time = self._compute_comm_time(submitter, task_idx)
            u_comm = self._compute_comm_energy(submitter, task_idx)
            total_time = comp_time + comm_time
            u_total_cost = u_comp + u_comm

        feasible = (phi_ji <= phi_j) and (total_time <= time_constraint)
        utility = (bi * phi_ji - u_total_cost) if feasible else 0.0

        return utility, total_time, feasible

    def compute_utility_matrix(self):
        """U[j,i] = utility of assigning task i to server j (for current tasks)."""
        U = np.zeros((self.num_servers, self.num_tasks), dtype=np.float32)
        for j in range(self.num_servers):
            for i in range(self.num_tasks):
                utility, _, _ = self._compute_utility(j, i, j)
                U[j, i] = utility
        return U

    def compute_soft_reward(self, raw_action):
        """Expected reward under per-task softmax assignment — differentiable proxy reward."""
        mat = np.asarray(raw_action, dtype=np.float64).reshape(self.num_servers, self.num_tasks)
        U = self.compute_utility_matrix()
        total = 0.0
        for i in range(self.num_tasks):
            col = mat[:, i]
            e = np.exp(col - col.max())
            probs = e / e.sum()
            total += float(np.dot(probs, U[:, i]))
        return total

    def step(self, action):
        action_matrix = action.reshape(self.num_servers, self.num_tasks)
        assignment = np.argmax(action_matrix, axis=0)

        total_utility = 0.0
        total_time = 0.0
        feasible_count = 0
        task_utilities = []
        task_times = []

        for task_idx in range(self.num_tasks):
            assigned_server = assignment[task_idx]
            utility, time_used, feasible = self._compute_utility(
                assigned_server, task_idx, assigned_server
            )
            total_utility += utility
            total_time += time_used
            task_utilities.append(utility)
            task_times.append(time_used)
            if feasible:
                feasible_count += 1

        self._generate_tasks()
        next_state = self._get_state()

        info = {
            "total_utility": total_utility,
            "total_time": total_time,
            "completed_tasks": feasible_count,
            "feasible_count": feasible_count,
            "assignment": assignment,
            "task_utilities": task_utilities,
            "task_times": task_times,
            "throughput": feasible_count / self.num_tasks,
            "avg_delay": total_time / self.num_tasks,
            "energy": sum(
                self._compute_comp_energy(assignment[i], i) for i in range(self.num_tasks)
            ),
            "hard_reward": total_utility,
        }

        done = True
        return next_state, total_utility, done, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._generate_tasks()
        return self._get_state()

    def get_server_resources(self):
        return self.servers

    def get_task_descriptions(self):
        return self.tasks
