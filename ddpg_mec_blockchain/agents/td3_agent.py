"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) — Fujimoto et al. 2018.
"""

import numpy as np
import torch
import torch.nn.functional as F
import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.actor import Actor
from networks.attention_actor import AttentionActor
from networks.critic import TwinCritic
from networks.replay_buffer import ReplayBuffer
from config import DEVICE, TD3_CONFIG


class TD3Agent:
    def __init__(self, state_dim, action_dim, num_servers=None, num_tasks=None, config=None):
        cfg = config or TD3_CONFIG
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.batch_size = cfg["batch_size"]
        self.policy_noise = cfg["policy_noise"]
        self.noise_clip = cfg["noise_clip"]
        self.policy_delay = cfg["policy_delay"]
        self.update_count = 0
        self.num_servers = num_servers
        self.num_tasks = num_tasks

        if num_servers is not None and num_tasks is not None:
            self.actor = AttentionActor(num_servers, num_tasks).to(DEVICE)
        else:
            self.actor = Actor(state_dim, action_dim, cfg["hidden_dims"]).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.twin_critic = TwinCritic(state_dim, action_dim, cfg["hidden_dims"]).to(DEVICE)
        self.twin_critic_target = copy.deepcopy(self.twin_critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.twin_critic.parameters(), lr=cfg["critic_lr"])

        self.buffer = ReplayBuffer(state_dim, action_dim, cfg["buffer_size"])

    def _per_task_softmax(self, logits):
        """Softmax over servers for each task independently."""
        batch = logits.shape[0]
        mat = logits.view(batch, self.num_servers, self.num_tasks)
        probs = F.softmax(mat, dim=1)
        return probs.view(batch, self.num_servers * self.num_tasks)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if not evaluate:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action = action + noise
        return action  # raw logits

    def store(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < 64:
            return None, None

        use_soft = self.num_servers is not None
        self.update_count += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        soft_actions = self._per_task_softmax(actions) if use_soft else actions.clamp(-1, 1)

        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            if use_soft:
                noise = (torch.randn_like(next_raw) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_raw = (next_raw + noise).clamp(-1, 1)
                next_soft = self._per_task_softmax(next_raw)
            else:
                noise = (torch.randn_like(next_raw) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_soft = (next_raw + noise).clamp(-1, 1)
            q1_target, q2_target = self.twin_critic_target(next_states, next_soft)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(q1_target, q2_target)

        q1, q2 = self.twin_critic(states, soft_actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.update_count % self.policy_delay == 0:
            raw_actor = self.actor(states)
            soft_actor = self._per_task_softmax(raw_actor) if use_soft else raw_actor
            actor_loss = -self.twin_critic.Q1(states, soft_actor).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.twin_critic.parameters(), self.twin_critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return actor_loss.item() if actor_loss is not None else 0.0, critic_loss.item()
