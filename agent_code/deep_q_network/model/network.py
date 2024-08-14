import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, channel_size: int, column_size: int, row_size: int, action_size: int) -> None:
        """ Initializes the deep Q-network.

        Args:
            state_size (int): The size of the input state.
            action_size (int): The number of actions.
        """
        super(Network, self).__init__()

        # NB: The internal grid (field, coins, etc.) is of size (column_size, row_size), which is the TRANSPOSE of the 
        #     grid that is drawn in the GUI. This is because the GUI draws the grid with the x-axis as the horizontal axis
        #     and the y-axis as the vertical axis, while the internal grid is stored with the x-axis as the vertical axis
        #     and the y-axis as the horizontal axis. This is not problematic as long as the internal grid is used consistently.
        self.layers = nn.Sequential(
            # Input: (batch_size, channel_size, column_size, row_size)
            # Output: (batch_size, 8, column_size, row_size)
            nn.Conv2d(in_channels=channel_size, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Input: (batch_size, 8, column_size, row_size)
            # Output: (batch_size, 16, column_size - 2, row_size - 2)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Input: (batch_size, 16, column_size - 2, row_size - 2)
            # Output: (batch_size, 16 * (column_size - 2) * (row_size - 2))
            nn.Flatten(),
            # Input: (batch_size, 6 * (column_size - 2) * (row_size - 2))
            # Output: (batch_size, 128)
            nn.Linear(16 * (column_size - 2) * (row_size - 2), 128),
            nn.ReLU(),
            # Input: (batch_size, 128)
            # Output: (batch_size, action_size)
            nn.Linear(128, action_size)
        )


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the deep Q-network.

        Args:
            state (torch.Tensor): The input state to the Q-network.

        Returns:
            torch.Tensor: The Q-values of the actions for the input state.
        """
        return self.layers(state)

    
    def initialize_weights_kaiming(self) -> None:
        """ Initialize the weights of the network using the Kaiming initialization. """
        
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)