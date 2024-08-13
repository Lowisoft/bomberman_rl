import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        """ Initializes the deep Q-network.

        Args:
            state_size (int): The size of the input state.
            action_size (int): The number of actions.
        """
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the deep Q-network.

        Args:
            state (torch.Tensor): The input state to the Q-network.

        Returns:
            torch.Tensor: The Q-values of the actions for the input state.
        """
        return self.layers(state)