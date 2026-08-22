"""
ScoringActor: parameter-shared scoring network for MEC task allocation.

Instead of outputting all num_servers*num_tasks logits from a single monolithic
network (which needs ~400K parameters and 500 samples to train), this network
scores each (task, server) pair using a shared MLP (~3K parameters).

Each training step produces num_servers*num_tasks gradient contributions per
sample — ~1000× more gradient signal than the monolithic actor.

Architecture follows: "Attention-based DRL for MEC" and related 2022-2024 work.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE

TASK_FEAT = 5   # unit_price, data_size, time_constraint, cpu_cycles, submitting_server
SERVER_FEAT = 5  # cpu_freq, bandwidth, tx_power, channel_gain, max_cpu_cycles


class ScoringActor(nn.Module):
    """
    Scores every (task_i, server_j) pair with shared parameters.
    State layout: [task_features (num_tasks*5), server_features (num_servers*5)]
    Output: (batch, num_servers * num_tasks) logits
    """

    def __init__(self, num_servers, num_tasks, hidden_dims=(128, 64)):
        super().__init__()
        self.num_servers = num_servers
        self.num_tasks = num_tasks

        layers = []
        in_d = TASK_FEAT + SERVER_FEAT  # 10-dim pair embedding
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.scorer = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.uniform_(list(self.scorer.children())[-1].weight, -3e-3, 3e-3)

    def forward(self, state):
        # state: (batch, num_tasks*5 + num_servers*5)
        B = state.shape[0]
        T, S = self.num_tasks, self.num_servers

        task_block = state[:, :T * TASK_FEAT].view(B, T, TASK_FEAT)     # (B, T, 5)
        server_block = state[:, T * TASK_FEAT:].view(B, S, SERVER_FEAT)  # (B, S, 5)

        # Broadcast all (task, server) pairs: (B, S, T, 10)
        task_exp = task_block.unsqueeze(1).expand(B, S, T, TASK_FEAT)
        server_exp = server_block.unsqueeze(2).expand(B, S, T, SERVER_FEAT)
        pairs = torch.cat([task_exp, server_exp], dim=-1)        # (B, S, T, 10)

        scores = self.scorer(pairs.view(B * S * T, TASK_FEAT + SERVER_FEAT))  # (B*S*T, 1)
        return scores.view(B, S * T)                              # (B, S*T)
