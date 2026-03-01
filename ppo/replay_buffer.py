_# -*- coding: utf-8 -*-
"""
Replay Buffer for On-Policy PPO.

This module provides a simple replay buffer tailored for on-policy algorithms
like PPO. Unlike off-policy buffers that sample randomly, this buffer stores
a complete rollout of experience and serves all of it for a single learning
update before being cleared.
"""

from typing import List, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """A buffer to store trajectories of experience for PPO.

    This buffer collects state, action, reward, next_state, done, log_prob, and
    action_mask tuples until it is full or the `get_all` method is called.
    It then converts the collected data into tensors for the learning step.

    Attributes:
        device (torch.device): The device (CPU or GPU) to store tensors on.
        states (List[np.ndarray]): List of state arrays.
        actions (List[int]): List of action indices.
        rewards (List[float]): List of rewards.
        next_states (List[np.ndarray]): List of next state arrays.
        dones (List[bool]): List of done flags.
        log_probs (List[float]): List of action log probabilities.
        action_masks (List[np.ndarray]): List of action mask arrays.
    """

    def __init__(self, capacity: int, device: torch.device):
        """Initializes the ReplayBuffer.

        Args:
            capacity: The maximum number of transitions to store (not strictly
                enforced for this on-policy buffer, but kept for consistency).
            device: The PyTorch device to move tensors to.
        """
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.action_masks = []

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob: float, action_mask: np.ndarray):
        """Adds a single transition to the buffer.

        Args:
            state: The environment state.
            action: The action taken.
            reward: The received reward.
            next_state: The resulting state.
            done: A flag indicating if the episode terminated.
            log_prob: The log probability of the action taken.
            action_mask: The action mask for the given state.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask)

    def get_all(self) -> Tuple[torch.Tensor, ...]:
        """Retrieves all stored transitions as tensors.

        Returns:
            A tuple of tensors containing all collected experience:
            (states, actions, rewards, next_states, dones, log_probs, action_masks)
        """
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        action_masks = torch.FloatTensor(np.array(self.action_masks)).to(self.device)

        return states, actions, rewards, next_states, dones, log_probs, action_masks

    def clear(self):
        """Clears all transitions from the buffer.

        This is called after each learning update in an on-policy setting.
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        """Returns the current number of transitions in the buffer.

        Returns:
            The number of stored transitions.
        """
        return len(self.states)
