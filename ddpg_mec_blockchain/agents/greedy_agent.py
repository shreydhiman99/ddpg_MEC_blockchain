"""Greedy baseline: assigns each task to the server with highest reward potential."""
import numpy as np


class GreedyAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, evaluate=False):
        n = self.env.num_servers
        m = self.env.num_tasks
        action_matrix = np.zeros((n, m))

        for task_idx in range(m):
            best_server = 0
            best_score = -np.inf
            for server_idx in range(n):
                bi = self.env.tasks["unit_price"][task_idx]
                phi = self.env.tasks["cpu_cycles_needed"][server_idx, task_idx]
                score = bi * phi
                if score > best_score:
                    best_score = score
                    best_server = server_idx
            action_matrix[best_server, task_idx] = 1.0

        return action_matrix.flatten()

    def store(self, *args): pass
    def update(self): return None, None
