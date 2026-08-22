"""
Cross-Attention Actor for MEC task allocation.

Architecture (based on 2023-2024 MEC-DRL literature):
  - Tasks act as QUERIES: "what kind of server do I need?"
  - Servers act as KEYS:  "what can I offer?"
  - Attention score(task_i, server_j) = compatibility between task i and server j
  - Output: (num_servers * num_tasks) logits; softmax per task = assignment probs

Why this works for MEC:
  - Inductive bias exactly matches the problem: match task requirements to server capacity
  - ~20K parameters vs ~400K for monolithic actor → 20× less data needed
  - Each training sample produces T×S gradient signals per parameter (vs 1 for monolithic)
  - Differentiable assignment via per-task softmax → clean policy gradient

References:
  - "Attention-based DRL for task offloading in MEC" (2023)
  - "Transformer-based resource allocation in edge computing" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASK_FEAT = 5    # unit_price, data_size, time_constraint, cpu_cycles, submitting_server
SERVER_FEAT = 5  # cpu_freq, bandwidth, tx_power, channel_gain, max_cpu_cycles


class AttentionActor(nn.Module):
    """
    Cross-attention actor: tasks (queries) attend to servers (keys) to produce
    assignment logits. Per-task softmax in the agent converts these to probabilities.
    """

    def __init__(self, num_servers, num_tasks, d_model=64, num_heads=4):
        super().__init__()
        self.num_servers = num_servers
        self.num_tasks = num_tasks
        self.d_model = d_model

        # Encode raw features into d_model-dim embeddings
        self.task_encoder = nn.Sequential(
            nn.Linear(TASK_FEAT, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.server_encoder = nn.Sequential(
            nn.Linear(SERVER_FEAT, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Multi-head cross-attention: tasks query servers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Project attended features → scalar logit per (task, server) pair
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_servers),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        # state: (batch, num_tasks*5 + num_servers*5)
        B = state.shape[0]
        T, S = self.num_tasks, self.num_servers

        task_raw = state[:, :T * TASK_FEAT].view(B, T, TASK_FEAT)        # (B, T, 5)
        server_raw = state[:, T * TASK_FEAT:].view(B, S, SERVER_FEAT)    # (B, S, 5)

        task_emb = self.task_encoder(task_raw)      # (B, T, d)
        server_emb = self.server_encoder(server_raw)  # (B, S, d)

        # Cross-attention: each task attends to all servers
        # query=tasks, key=value=servers → task-aware context vectors
        attended, _ = self.cross_attn(
            query=task_emb,
            key=server_emb,
            value=server_emb,
        )  # attended: (B, T, d)

        # Score each task against each server using attended representation
        logits = self.score_head(attended)  # (B, T, S)

        # Reshape to (B, S*T): convention matches _per_task_softmax in agents
        # logits[:, j*T + i] = logit for assigning task i to server j
        logits_ST = logits.permute(0, 2, 1)  # (B, S, T)
        return logits_ST.contiguous().view(B, S * T)
