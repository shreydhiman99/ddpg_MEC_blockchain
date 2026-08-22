"""
DDPG Agent — Lillicrap et al. (2015).
Implements Algorithm 2 from the paper.
"""

import numpy as np
import torch
import torch.nn.functional as F
import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.actor import Actor
from networks.attention_actor import AttentionActor
from networks.critic import Critic
from networks.replay_buffer import ReplayBuffer
from config import DEVICE, DDPG_CONFIG


class OUNoise:
    """Ornstein-Uhlenbeck process noise for exploration."""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


class DDPGAgent:
    def __init__(self, state_dim, action_dim, num_servers=None, num_tasks=None, config=None):
        cfg = config or DDPG_CONFIG
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.batch_size = cfg["batch_size"]
        self.warmup_steps = cfg["warmup_steps"]
        self.total_steps = 0
        self.num_servers = num_servers
        self.num_tasks = num_tasks

        if num_servers is not None and num_tasks is not None:
            self.actor = AttentionActor(num_servers, num_tasks).to(DEVICE)
        else:
            self.actor = Actor(state_dim, action_dim, cfg["hidden_dims"]).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, cfg["hidden_dims"]).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

        self.buffer = ReplayBuffer(state_dim, action_dim, cfg["buffer_size"])
        self.noise = OUNoise(action_dim, sigma=cfg["noise_std"])

        self.updates_per_step = cfg.get("updates_per_step", 1)
        self.actor_losses = []
        self.critic_losses = []

    def _per_task_softmax(self, logits):
        """Softmax over servers for each task independently. logits: (batch, S*T) or (S*T,)"""
        batch = logits.shape[0]
        mat = logits.view(batch, self.num_servers, self.num_tasks)
        probs = F.softmax(mat, dim=1)
        return probs.view(batch, self.num_servers * self.num_tasks)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if not evaluate:
            action = action + self.noise.sample()
        return action  # raw logits; caller applies softmax for soft_reward

    def store(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    def _update_once(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        use_soft = self.num_servers is not None

        # Convert stored raw logits to soft probabilities for critic input
        soft_actions = self._per_task_softmax(actions) if use_soft else actions.clamp(-1, 1)

        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            next_soft = self._per_task_softmax(next_raw) if use_soft else next_raw.clamp(-1, 1)
            target_q = self.critic_target(next_states, next_soft)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, soft_actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        raw_actor = self.actor(states)
        soft_actor = self._per_task_softmax(raw_actor) if use_soft else raw_actor
        actor_loss = -self.critic(states, soft_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def update(self):
        if len(self.buffer) < self.warmup_steps:
            return None, None

        al, cl = None, None
        for _ in range(self.updates_per_step):
            al, cl = self._update_once()

        self.actor_losses.append(al)
        self.critic_losses.append(cl)
        return al, cl

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
