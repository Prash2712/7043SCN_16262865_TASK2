_# -*- coding: utf-8 -*-
"""
Actor-Critic Model Architectures for the PPO Agent.

This module defines the neural network architectures for the Actor (policy)
and Critic (value function) components of the Proximal Policy Optimization (PPO)
algorithm. Both networks are implemented as simple Multi-Layer Perceptrons (MLPs)
using PyTorch.

- **Actor**: Takes the environment state as input and outputs a probability
  distribution over the discrete action space.
- **Critic**: Takes the environment state as input and outputs a single scalar
  value representing the estimated value of that state.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """The policy network (Actor) for the PPO agent.

    This network learns a policy by mapping states to a probability distribution
    over actions. An action mask is applied to the output logits to prevent the
    selection of invalid actions in the current game state.

    Attributes:
        state_dim (int): The dimensionality of the input state space.
        action_dim (int): The dimensionality of the output action space.
        layer1 (nn.Linear): The first fully connected layer.
        layer2 (nn.Linear): The second fully connected layer.
        output_layer (nn.Linear): The final layer producing action logits.
    """

    def __init__(self, state_dim: int, action_dim: int):
        """Initializes the Actor network.

        Args:
            state_dim: The size of the state vector from the environment.
            action_dim: The total number of possible actions.
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the network to get action probabilities.

        Args:
            state: A tensor representing the current environment state.
            action_mask: A binary tensor where `1` indicates a valid action
                and `0` indicates an invalid action.

        Returns:
            A tuple containing:
            - The logits (raw scores) for each action.
            - The log-probabilities of the action distribution.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        logits = self.output_layer(x)

        # Apply the action mask
        # Set logits of invalid actions to a very small number (-infinity)
        # so they have zero probability after softmax.
        masked_logits = logits.masked_fill(action_mask == 0, -1e9)

        # Create a probability distribution and calculate log-probabilities
        dist = Categorical(logits=masked_logits)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        return dist, log_probs


class Critic(nn.Module):
    """The value function network (Critic) for the PPO agent.

    This network learns to estimate the value of a given state, which is used
    to calculate the advantage function (how much better an action is than the
    average action from that state).

    Attributes:
        state_dim (int): The dimensionality of the input state space.
        layer1 (nn.Linear): The first fully connected layer.
        layer2 (nn.Linear): The second fully connected layer.
        output_layer (nn.Linear): The final layer producing the state value.
    """

    def __init__(self, state_dim: int):
        """Initializes the Critic network.

        Args:
            state_dim: The size of the state vector from the environment.
        """
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1)  # Outputs a single value

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass to estimate the value of the state.

        Args:
            state: A tensor representing the current environment state.

        Returns:
            A tensor containing the single scalar value of the state.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        state_value = self.output_layer(x)
        return state_value
