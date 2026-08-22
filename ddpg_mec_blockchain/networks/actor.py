import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


class Actor(nn.Module):
    """Actor network: maps state → action (continuous allocation scores)."""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)

    def forward(self, state):
        return self.net(state)
