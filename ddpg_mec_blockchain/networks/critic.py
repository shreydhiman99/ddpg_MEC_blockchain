import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


class Critic(nn.Module):
    """Critic network: maps (state, action) → Q-value."""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.out]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat([x, action], dim=-1)
        x = torch.relu(self.fc2(x))
        return self.out(x)


class TwinCritic(nn.Module):
    """Twin critics for TD3 — reduces overestimation bias."""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.critic1 = Critic(state_dim, action_dim, hidden_dims)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims)

    def forward(self, state, action):
        return self.critic1(state, action), self.critic2(state, action)

    def Q1(self, state, action):
        return self.critic1(state, action)
