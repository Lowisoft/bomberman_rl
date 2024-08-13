import random
import torch
from collections import deque
import numpy as np
from typing import Union, List
from utils import action_str_to_index


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
            next_state (Union[np.ndarray, None]): The next state of the environment. None if the game has ended.
        """

        # Add the experience to the buffer
        self.buffer.append((state, action_str_to_index(action), reward, next_state))
    

    def sample(self, batch_size: int) -> List[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Samples a batch of experiences from the buffer.

        Args: 
            batch_size (int): The number of experiences to sample into the batch.

           
        Returns:
            List[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A list containing the states, actions, rewards, next states and dones.
        """

        # Sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Extract the components of the experiences into separate lists
        states, actions, rewards, next_states = map(list, zip(*batch))
        # Stack the states to obtain the shape (batch_size, column_size, row_size, channel_size) and convert to tensor
        states = torch.from_numpy(np.stack(states, axis=0)).float().to(self.device)
        # Convert the actions to a tensor
        actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        # Convert the rewards to a tensor
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        # Create dones mask to indicate if the next state is None (i.e. the game has ended)
        dones = np.zeros(batch_size, dtype=bool)
        # Search for next states that are None, convert them to a dummy zero state and set the corresponding dones mask to 1
        for i, next_state in enumerate(next_states):
            if next_state is None:
                next_states[i] = np.zeros_like(states[0])
                dones[i] = 1
        # Stack the next states to obtain the shape (batch_size, column_size, row_size, channel_size) and convert to tensor
        next_states = torch.from_numpy(np.stack(next_states, axis=0)).float().to(self.device)
        # Convert the dones mask to a tensor
        dones = torch.from_numpy(dones).bool().to(self.device)

        return states, actions, rewards, next_states, dones