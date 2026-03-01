_# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization (PPO) Agent for Chef's Hat Gym.

This module contains the core PPO agent class, which is responsible for
interacting with the environment, collecting experience, and updating its
policy and value networks. It integrates with a reward shaper to allow for
dense, domain-knowledge-based rewards.
"""

import os
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent # Assumes ChefsHatGYM is in PYTHONPATH
from ppo.model import Actor, Critic
from ppo.replay_buffer import ReplayBuffer
from reward_shaping.shapers import BaseRewardShaper


class PPOAgent(BaseAgent):
    """A PPO agent that learns to play Chef's Hat.

    This agent implements the Proximal Policy Optimization algorithm. It uses an
    Actor-Critic architecture and a replay buffer to store and learn from
    trajectories of experience.

    Attributes:
        device (torch.device): The device (CPU or GPU) to run the networks on.
        actor (Actor): The policy network.
        critic (Critic): The value function network.
        optimizer_actor (optim.Adam): The optimizer for the Actor network.
        optimizer_critic (optim.Adam): The optimizer for the Critic network.
        reward_shaper (BaseRewardShaper): The reward shaping module.
        p_buffer (ReplayBuffer): The replay buffer for storing experience.
        p_hyperparameters (Dict[str, Any]): Dictionary of PPO hyperparameters.
    """

    def __init__(self, state_dim: int, action_dim: int, p_hyperparameters: Dict[str, Any], reward_shaper: BaseRewardShaper):
        """Initializes the PPOAgent.

        Args:
            state_dim: The dimensionality of the state space.
            action_dim: The dimensionality of the action space.
            p_hyperparameters: A dictionary of hyperparameters for the PPO algorithm.
            reward_shaper: An initialized reward shaper object.
        """
        super(PPOAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPOAgent] Using device: {self.device}")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=p_hyperparameters["learning_rate"])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=p_hyperparameters["learning_rate"])

        self.reward_shaper = reward_shaper
        self.p_buffer = ReplayBuffer(p_hyperparameters["batch_size"], self.device)
        self.p_hyperparameters = p_hyperparameters

        # Trackers for logging
        self.actor_loss_history = []
        self.critic_loss_history = []

    def request_action(self, observation: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Requests an action from the agent based on the current observation.

        Args:
            observation: A dictionary containing the current game state, including
                'state' and 'action_mask'.

        Returns:
            A tuple containing:
            - The selected action index.
            - A dictionary with auxiliary information (log probability).
        """
        state = torch.FloatTensor(observation["state"]).to(self.device)
        action_mask = torch.FloatTensor(observation["action_mask"]).to(self.device)

        with torch.no_grad():
            dist, _ = self.actor(state, action_mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), {"log_prob": log_prob.item()}

    def update_replay_buffer(self, transition: Tuple):
        """Adds a new transition to the replay buffer.

        Args:
            transition: A tuple containing (state, action, reward, next_state, done, log_prob, action_mask).
        """
        state, action, reward, next_state, done, log_prob, action_mask = transition

        # Apply reward shaping
        shaped_reward = self.reward_shaper.shape(state, reward, done, {})

        self.p_buffer.add(
            state["state"],
            action,
            shaped_reward,
            next_state["state"],
            done,
            log_prob,
            state["action_mask"]
        )

    def learn(self):
        """Updates the Actor and Critic networks using the collected experience.

        This method implements the core PPO learning logic, including calculating
        advantages, computing policy and value losses, and performing backpropagation.
        """
        states, actions, rewards, next_states, dones, log_probs, action_masks = self.p_buffer.get_all()

        # --- Calculate Advantages using GAE ---
        with torch.no_grad():
            state_values = self.critic(states).squeeze()
            next_state_values = self.critic(next_states).squeeze()

        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.p_hyperparameters["gamma"] * next_state_values[t] * (1 - dones[t]) - state_values[t]
            advantages[t] = last_advantage = delta + self.p_hyperparameters["gamma"] * self.p_hyperparameters["gae_lambda"] * (1 - dones[t]) * last_advantage

        returns = advantages + state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Perform PPO Update for multiple epochs ---
        for _ in range(self.p_hyperparameters["epochs"]):
            # --- Actor (Policy) Loss ---
            dist, _ = self.actor(states, action_masks)
            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.p_hyperparameters["clip_epsilon"], 1 + self.p_hyperparameters["clip_epsilon"]) * advantages
            entropy_bonus = dist.entropy().mean()

            actor_loss = -torch.min(surr1, surr2).mean() - self.p_hyperparameters["entropy_coef"] * entropy_bonus

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.p_hyperparameters["max_grad_norm"])
            self.optimizer_actor.step()

            # --- Critic (Value) Loss ---
            new_state_values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(new_state_values, returns)

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.p_hyperparameters["max_grad_norm"])
            self.optimizer_critic.step()

        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

        self.p_buffer.clear()

    def save_models(self, directory: str, name: str):
        """Saves the Actor and Critic model weights to files.

        Args:
            directory: The directory to save the models in.
            name: A descriptive name to include in the filename (e.g., 'final').
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"actor_{name}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"critic_{name}.pt"))
        print(f"[PPOAgent] Models saved to {directory}")

    def load_models(self, actor_path: str, critic_path: str):
        """Loads pre-trained Actor and Critic model weights.

        Args:
            actor_path: The file path to the saved Actor weights.
            critic_path: The file path to the saved Critic weights.
        """
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor.eval()
        self.critic.eval()
        print(f"[PPOAgent] Models loaded from {actor_path} and {critic_path}")
