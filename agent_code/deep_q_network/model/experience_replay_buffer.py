import random
import pickle
import torch
from collections import deque
import numpy as np
import torchvision.transforms.functional as F
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
        # Get the shape of the states (will used later to create the dummy zero state)
        state_shape = states[0].shape
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
            np.zeros(state_shape) if done else next_state
            for next_state, done in zip(next_states, dones)
        ])
        # Stack the next states to obtain the shape (batch_size, channel_size, column_size, row_size) and convert to tensor
        next_states = torch.from_numpy(np.stack(next_states, axis=0)).float().to(self.device)
        # Convert the dones mask to a tensor
        # NB: The float data type is used because the dones mask is used in a calculation with rewards (which are float)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Generate random masks for each transformation
        horizontal_flip_mask = torch.rand(batch_size, device=self.device) < 0.5
        vertical_flip_mask = torch.rand(batch_size, device=self.device) < 0.5
        rotation_mask = torch.rand(batch_size, device=self.device) < 0.5

        # Detect whether the action is a move (not the case for BOMB or WAIT)
        move_mask = (actions <= 4).to(self.device)

        # Apply random horizontal flips
        if horizontal_flip_mask.any():
            # Horizontally flip the states
            states[horizontal_flip_mask] = F.hflip(states[horizontal_flip_mask])
            # Horizontally flip the next states
            next_states[horizontal_flip_mask] = F.hflip(next_states[horizontal_flip_mask])
            # Horizontal flip the actions
            # NB: We should only swap LEFT and RIGHT, which are both odd (1 or 3 resp.)
            #     Thus we can filter for LEFT and RIGHT with (actions[horizontal_flip_mask] % 2)
            actions[horizontal_flip_mask & move_mask] = (actions[horizontal_flip_mask & move_mask] + ((actions[horizontal_flip_mask & move_mask] % 2) * 2)) % 4

        # Apply random vertical flips
        if vertical_flip_mask.any():
            # Vertically flip the states
            states[vertical_flip_mask] = F.vflip(states[vertical_flip_mask])
            # Vertically flip the next states
            next_states[vertical_flip_mask] = F.vflip(next_states[vertical_flip_mask])
            # Horizontal flip the actions
            # NB: We should only swap UP and DOWN, which are both even (0 or 2 resp.)
            #     Thus we can filter for UP and DOWN with ((1 + actions[vertical_flip_mask]) % 2)
            actions[vertical_flip_mask & move_mask] = (actions[vertical_flip_mask & move_mask] + (((1 + actions[vertical_flip_mask & move_mask]) % 2) * 2)) % 4

        # Apply random rotations
        # IMPORTANT: The rotations are performed counterclockwise
        if rotation_mask.any():
            # Randomly draw the rotation multiplicities (either 1, 2 or 4)
            rotation_multip = torch.randint(1, 4, size=(batch_size, ), device=self.device)
            # Rotate the states         
            states = torch.stack([
                torch.rot90(state, k=multip.item(), dims=[1, 2]) if mask else state
                for state, mask, multip in zip(states, rotation_mask, rotation_multip)
            ])
             # Rotate the next_states    
            next_states = torch.stack([
                torch.rot90(next_state, k=multip.item(), dims=[1, 2]) if mask else next_state
                for next_state, mask, multip in zip(next_states, rotation_mask, rotation_multip)
            ])
            # Rotate the actions
            actions[rotation_mask & move_mask] = (actions[rotation_mask & move_mask] - rotation_multip[rotation_mask & move_mask]) % 4

        return states, actions, rewards, next_states, dones


    def save(self, path: str) -> None:
        """ Saves the buffer to a file.

        Args:
            path (str): The path to the file where the buffer will be saved.
        """

        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)


    def load(self, path: str) -> None:
        """ Loads the buffer from a file.

        Args:
            path (str): The path to the file where the buffer will be loaded from.
        """

        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)