"""
Q-Learning baseline with discretized state-action space.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QLEARNING_CONFIG


class QLearningAgent:
    def __init__(self, state_dim, action_dim, num_servers, num_tasks, config=None):
        cfg = config or QLEARNING_CONFIG
        self.alpha = cfg["alpha"]
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon"]
        self.num_servers = num_servers
        self.num_tasks = num_tasks
        self.action_dim = action_dim
        self.n_state_bins = 10
        self.n_action_bins = num_servers
        # Cap action space to avoid memory explosion at large server counts
        n_action_cols = min(num_servers ** min(num_tasks, 3), 50000)
        self.q_table = np.zeros((self.n_state_bins ** 3, n_action_cols))

    def _discretize_state(self, state):
        bins = np.linspace(0, 1, self.n_state_bins)
        idx = np.digitize(state[:3], bins) - 1
        return int(sum(i * (self.n_state_bins ** j) for j, i in enumerate(idx))) % self.q_table.shape[0]

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.uniform(-1, 1, size=self.action_dim)
        state_idx = self._discretize_state(state)
        best_a = int(np.argmax(self.q_table[state_idx]))
        action = np.zeros(self.action_dim)
        for task_idx in range(self.num_tasks):
            server = best_a % self.num_servers
            action[server * self.num_tasks + task_idx] = 1.0
            best_a //= self.num_servers
        return action

    def store(self, state, action, reward, next_state, done):
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        a = int(np.sum(np.arange(len(action)) * (action > 0))) % self.q_table.shape[1]
        current_q = self.q_table[s, a]
        target_q = reward + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.alpha * (target_q - current_q)

    def update(self): return None, None
