"""Random baseline: assigns tasks randomly to servers."""
import numpy as np


class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state, evaluate=False):
        return np.random.uniform(-1, 1, size=self.action_dim)

    def store(self, *args): pass
    def update(self): return None, None
