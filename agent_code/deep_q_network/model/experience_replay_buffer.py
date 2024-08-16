import random
import torch
from collections import deque
import numpy as np
from typing import Union, Tuple
from ..utils import action_str_to_index


class ExperienceReplayBuffer(object):
    def __init__(self, buffer_capacity: int, device: torch.device) -> None:
        """ Initializes the experience replay buffer.

        Args:
            buffer_capacity (int): The maximum number of experiences that can be stored in the buffer.
            device (torch.device): The device to be used.
        """

        # Initialize the buffer
        self.buffer = deque(maxlen=buffer_capacity)
        # Set the device to be used
        self.device = device


    def __len__(self) -> int:
        """ Returns the number of experiences in the buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.buffer)
        

    def push(self, state: np.ndarray, action: str, reward: float, next_state: Union[np.ndarray, None]) -> None:
        """ Pushes a new experience to the buffer.

        Args:
            state (np.ndarray): The current state of the environment.
            action (str): The action taken in the state.
            reward (float): The reward received after taking the action.
            next_state (Union[np.ndarray, None]): The next state of the environment. None if the round/game has ended by death of the agent.
        """

        # Add the experience to the buffer
        self.buffer.append((state, action_str_to_index(action), reward, next_state))
    

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Samples a batch of experiences from the buffer.

        Args: 
            batch_size (int): The number of experiences to sample into the batch.

           
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A list containing the states, actions, rewards, next states and dones.
        """

        # Sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Extract the components of the experiences into separate lists
        states, actions, rewards, next_states = map(list, zip(*batch))
        # Stack the states to obtain the shape (batch_size, channel_size, column_size, row_size) and convert to tensor
        states = torch.from_numpy(np.stack(states, axis=0)).float().to(self.device)
        # Convert the actions to a tensor
        # NB: The long data type is used because the actions are used as indices (which need to be long in PyTorch)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        # Convert the rewards to a tensor
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        # Create dones mask to indicate if the next state is None (i.e. the game has ended)
        dones = np.array([next_state is None for next_state in next_states], dtype=bool)
        # Search for next states that are None and convert them to a dummy zero state
        next_states = np.array([
            np.zeros_like(states[0]) if done else next_state
            for next_state, done in zip(next_states, dones)
        ])
        # Stack the next states to obtain the shape (batch_size, channel_size, column_size, row_size) and convert to tensor
        next_states = torch.from_numpy(np.stack(next_states, axis=0)).float().to(self.device)
        # Convert the dones mask to a tensor
        # NB: The float data type is used because the dones mask is used in a calculation with rewards (which are float)
        dones = torch.from_numpy(dones).float().to(self.device)

        return states, actions, rewards, next_states, dones