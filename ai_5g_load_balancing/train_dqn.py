"""
DQN training loop for the load-balancing environment.

This script keeps things lightweight so you can iterate quickly:
- shared policy across all UEs (each state/action pair is one replay entry)
- epsilon-greedy exploration
- experience replay + target network
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import NetworkEnvironment
from models import BaseStation, UserEquipment
from rl_models import QNetwork


# ---------------------------------------------------------------------------
# Environment factory (mirrors main.py)
# ---------------------------------------------------------------------------

def build_network():
    base_stations = [
        BaseStation(0, 20, 20, tier="macro"),
        BaseStation(1, 80, 20, tier="macro"),
        BaseStation(2, 50, 80, tier="macro"),
        BaseStation(3, 35, 55, tier="micro"),
        BaseStation(4, 70, 60, tier="micro"),
    ]
    users = [
        UserEquipment(i, x=(i * 7) % 100, y=(i * 13) % 100) for i in range(40)
    ]
    return base_stations, users


# ---------------------------------------------------------------------------
# Training config + utilities
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    episodes: int = 200
    steps_per_episode: int = 60
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    min_buffer: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update: int = 25
    seed: int = 42
    output_path: str = "artifacts/dqn_weights.pt"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).long(),
            torch.from_numpy(rewards).float(),
            torch.from_numpy(next_states).float(),
            torch.from_numpy(dones.astype(np.float32)).float(),
        )


class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg: TrainConfig, device=None):
        self.cfg = cfg
        self.action_dim = action_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss()
        self.epsilon = cfg.epsilon_start

    def select_actions(self, states: np.ndarray, candidate_map: Sequence[Sequence[int]]):
        actions = []
        if random.random() < self.epsilon:
            for candidates in candidate_map:
                actions.append(random.randrange(max(len(candidates), 1)))
            return actions

        state_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        q_values = q_values.cpu().numpy()
        for i, candidates in enumerate(candidate_map):
            if len(candidates) == 0:
                actions.append(0)
            else:
                best_idx = int(np.argmax(q_values[i, : len(candidates)]))
                actions.append(best_idx)
        return actions

    def update_epsilon(self):
        self.epsilon = max(
            self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay
        )

    def learn(self, buffer: ReplayBuffer):
        if len(buffer) < max(self.cfg.batch_size, self.cfg.min_buffer):
            return None
        states, actions, rewards, next_states, dones = buffer.sample(
            self.cfg.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next = self.target_net(next_states).max(1)[0]
            targets = rewards + self.cfg.gamma * max_next * (1 - dones)
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    bs, users = build_network()
    env = NetworkEnvironment(bs, users)
    states, candidate_map = env.reset_for_rl()
    state_dim = states.shape[1]
    action_dim = env.rl_config.get("num_candidates", 3)
    agent = DQNAgent(state_dim, action_dim, cfg)
    buffer = ReplayBuffer(cfg.buffer_size)

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

    for episode in range(1, cfg.episodes + 1):
        states, candidate_map = env.reset_for_rl()
        episode_reward = 0.0
        loss_val = None

        for step in range(cfg.steps_per_episode):
            actions = agent.select_actions(states, candidate_map)
            next_states, rewards, info = env.step_with_actions(
                actions, candidate_map=candidate_map, update_history=False
            )
            next_candidate_map = info["candidate_map"]

            dones = np.zeros_like(rewards, dtype=np.float32)
            if step == cfg.steps_per_episode - 1:
                dones[:] = 1.0

            for idx in range(len(rewards)):
                buffer.push(
                    states[idx],
                    actions[idx],
                    rewards[idx],
                    next_states[idx],
                    bool(dones[idx]),
                )

            loss_val = agent.learn(buffer)
            agent.update_epsilon()
            states = next_states
            candidate_map = next_candidate_map
            episode_reward += float(np.mean(rewards))

        if episode % cfg.target_update == 0:
            agent.update_target()

        print(
            f"Episode {episode}/{cfg.episodes} :: "
            f"avg reward={episode_reward:.3f} epsilon={agent.epsilon:.3f} "
            f"buffer={len(buffer)} loss={'{:.4f}'.format(loss_val) if loss_val else 'n/a'}"
        )

    torch.save(agent.policy_net.state_dict(), cfg.output_path)
    print(f"Saved trained weights to {cfg.output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN load-balancing agent")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--min-buffer", type=int, default=2_000)
    parser.add_argument("--target-update", type=int, default=25)
    parser.add_argument("--output", type=str, default="artifacts/dqn_weights.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        target_update=args.target_update,
        output_path=args.output,
    )
    train(cfg)


if __name__ == "__main__":
    main()
