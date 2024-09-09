import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, channel_size: int, column_size: int, row_size: int, action_size: int, hidden_layer_size: int, add_state_size: int, use_dueling_dqn: bool = False) -> None:
        """ Initializes the deep Q-network.

        Args:
            channel_size (int): The number of input channels.  
            state_size (int): The size of the input state.
            action_size (int): The number of actions.
            hidden_layer_size (int): Size of the hidden layer.
            use_dueling_dqn (bool): Whether to use the dueling DQN architecture.
        """
        super(Network, self).__init__()

        # NB: The internal grid (field, coins, etc.) is of size (column_size, row_size), which is the TRANSPOSE of the 
        #     grid that is drawn in the GUI. This is because the GUI draws the grid with the x-axis as the horizontal axis
        #     and the y-axis as the vertical axis, while the internal grid is stored with the x-axis as the vertical axis
        #     and the y-axis as the horizontal axis. This is not problematic as long as the internal grid is used consistently.

        # Store whether the dueling DQN architecture is used
        self.use_dueling_dqn = use_dueling_dqn

        # Calculate the size of the flattened feature map after the convolutional layers
        self.feature_size = 16 * (column_size - 2) * (row_size - 2)

        # Calculate the size of the input to the fully connected layer
        self.input_size = self.feature_size + add_state_size

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
              # Input: (batch_size, channel_size, column_size, row_size)
                # Output: (batch_size, 8, column_size, row_size)
                nn.Conv2d(in_channels=channel_size, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Input: (batch_size, 8, column_size, row_size)
                # Output: (batch_size, 16, column_size - 2, row_size - 2)
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                # Input: (batch_size, 16, column_size - 2, row_size - 2)
                # Output: (batch_size, self.feature_size)
                nn.Flatten(),
        )

        if self.use_dueling_dqn:
            # Define the advantage layers
            self.advantage_layers = nn.Sequential(
                # Input: (batch_size, self.input_size)
                # Output: (batch_size, hidden_layer_size)
                nn.Linear(self.input_size, hidden_layer_size),
                nn.ReLU(),
                # Input: (batch_size, hidden_layer_size)
                # Output: (batch_size, action_size)
                nn.Linear(hidden_layer_size, action_size)
            )
            # Define the value layers
            self.value_layers = nn.Sequential(
                # Input: (batch_size, self.input_size)
                # Output: (batch_size, hidden_layer_size)
                nn.Linear(self.input_size, hidden_layer_size),
                nn.ReLU(),
                # Input: (batch_size, hidden_layer_size)
                # Output: (batch_size, 1)
                nn.Linear(hidden_layer_size, 1)
            )
        else:
            self.layers = nn.Sequential(
                # Input: (batch_size, self.input_size)
                # Output: (batch_size, hidden_layer_size)
                nn.Linear(self.input_size, hidden_layer_size),
                nn.ReLU(),
                # Input: (batch_size, hidden_layer_size)
                # Output: (batch_size, action_size)
                nn.Linear(hidden_layer_size, action_size)
            )


    def forward(self, state: torch.Tensor, add_state: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the deep Q-network.

        Args:
            state (torch.Tensor): The input state to the Q-network.
            add_state (torch.Tensor): The additional input state to the Q-network.

        Returns:
            torch.Tensor: The Q-values of the actions for the input state.
        """

        # Forward pass through the convolutional layers
        flattened_conv = self.conv_layers(state)
        # Concatenate the flattened convolutional layers with the additional state
        combined_input = torch.cat((flattened_conv, add_state), dim=1)

        # Check if the dueling DQN architecture is used
        if not self.use_dueling_dqn:
            # If not, return the output of the layers
            return self.layers(combined_input)
        else:
            # Otherwise, return the output of the advantage and value layers (combined)
            advantage = self.advantage_layers(combined_input)
            value = self.value_layers(combined_input)
            return value + advantage - advantage.mean()

    
    def initialize_weights_kaiming(self) -> None:
        """ Initialize the weights of the network using the Kaiming initialization. """
        
        # Define the initialization function
        def init_weights(m):
            # Check if the module is a convolutional or linear layer
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # If so use the Kaiming initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Check if the module has a bias
                if m.bias is not None:
                    # If so, initialize the bias to zero
                    nn.init.constant_(m.bias, 0)

        # Apply the initialization function to the network
        self.apply(init_weights)