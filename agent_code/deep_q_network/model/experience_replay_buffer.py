import random
import pickle
import torch
from collections import deque
import numpy as np
import torchvision.transforms.functional as F
from typing import Union, Tuple
from ..utils import action_str_to_index


class ExperienceReplayBuffer(object):
    def __init__(self, buffer_capacity: int, device: torch.device, discount_rate: float, transform_batch_randomly: bool = False, n_steps: int = 1) -> None:
        """ Initializes the experience replay buffer.

        Args:
            buffer_capacity (int): The maximum number of experiences that can be stored in the buffer.
            device (torch.device): The device to be used.
            discount_rate (float): The discount rate. Only used if n_steps is greater than 1, i.e. a larger temporal differencing should be used
            transform_batch_randomly (bool, optional):  Whether the sampled batch should be transformed randomly. Default is False.
            n_steps (int, optional): The number of steps used when computing the temporal difference. Default is 1.
        """

        # Initialize the buffer
        self.buffer = deque(maxlen=buffer_capacity)
        # Set the device to be used
        self.device = device
        # Whether the sampled batch should be transformed randomly
        self.transform_batch_randomly = transform_batch_randomly
        # The number of steps used when computing the temporal difference
        self.n_steps = n_steps
        # Check if a larger temporal differencing should be used
        if self.n_steps > 1:
            # Set the discount rate
            self.discount_rate = discount_rate
            # Initialize the temporary buffer
            self.temporary_buffer = deque(maxlen=self.n_steps)


    def __len__(self) -> int:
        """ Returns the number of experiences in the buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.buffer)


    # def get_last_added_elem(self) -> Tuple[np.ndarray, str, float, np.ndarray]:
    #     """ Returns the most last added element of the buffer.
    #     Returns:
    #         Tuple[np.ndarray, str, float, np.ndarray]: The most last added element of the buffer.
    #     """
    #     return self.buffer[-1] 
        

    def push(self, state: np.ndarray, action: str, reward: float, next_state: Union[np.ndarray, None]) -> None:
        """ Pushes a new experience to the buffer.

        Args:
            state (np.ndarray): The current state of the environment.
            action (str): The action taken in the state.
            reward (float): The reward received after taking the action.
            next_state (Union[np.ndarray, None]): The next state of the environment. None if the round/game has ended.
        """
        if self.n_steps <= 1:
            # Add the experience to the buffer
            self.buffer.append((state, action_str_to_index(action), reward, next_state))
        else:
            # Push the current experience to the temporary buffer
            self.temporary_buffer.append((state, action_str_to_index(action), reward, next_state))
            # Check if the round/game finished
            if next_state is None: 
                # If so, clear the temporary buffer by adding the remaining experiences in the temporary buffer
                while len(self.temporary_buffer) > 0:
                    # Calculate the cumulative reward and append it to the buffer
                    self.compute_cumulative_reward_and_add_to_buffer()
                    # Remove the oldest experience
                    self.temporary_buffer.popleft()
            # Otherwise, if we have n experiences, calculate the cumulative reward of the oldest experience and append it to the buffer
            elif len(self.temporary_buffer) == self.n_steps:
                self.compute_cumulative_reward_and_add_to_buffer()  
                # Remove the oldest experience
                self.temporary_buffer.popleft()


    def compute_cumulative_reward_and_add_to_buffer(self) -> None:
        """ Computes the cumulative reward of the oldest experience in the temporary buffer
            and adds it to the main buffer.
        """
        
        # Compute the cumulative reward over the temporal buffer
        cumulative_reward = 0.0
        discount = 1.0
        for i in range(len(self.temporary_buffer)):
            cumulative_reward += discount * self.temporary_buffer[i][2] 
            discount *= self.discount_rate
 
        # Store the experience with the cumulative reward in the replay buffer
        self.buffer.append((
            self.temporary_buffer[0][0], # State of the oldest experience in the temporary buffer
            self.temporary_buffer[0][1], # Action of the oldest experience in the temporary buffer
            cumulative_reward, # Cumulative reward over the entire temporary buffer 
            self.temporary_buffer[-1][3], # Next state of the youngest experience in the buffer
        ))   


    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Samples a batch of experiences from the buffer.
            The every experience in the batch is randomly horizontally/vertically flipped and radomly rotated.

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

        if self.transform_batch_randomly:
            # Generate random masks for each transformation
            # NB: There are 8 different transformed states (cf. dihedral group D_4):
            #     1) identity 2) rotate90         3) rotate180         4) rotate270
            #     5) hflip    6) hflip + rotate90 7) hflip + rotate180 8) hflip + rotate270
            # NB2: vflip is the same as hflip + rotate180 and can thus be omitted
            horizontal_flip_mask = torch.rand(batch_size, device=self.device) < 0.5
            # Rotate with a probability of 75%. If so, the rotation angle (90, 180, 270) is uniformly sampled.
            # This ensures that no rotation, rotate90, rotate180 and rotate270 have all the same probability of 25%.
            rotation_mask = torch.rand(batch_size, device=self.device) < 0.75

            # Generate a mask indicating whether the action is a move (not the case for WAIT [4] or BOMB [5])
            move_mask = (actions < 4).to(self.device)

            # # NB: Uncomment these variables for assertions
            # old_states = states.detach().clone()
            # old_next_states = next_states.detach().clone()
            # old_actions = actions.detach().clone()

            # Apply random horizontal flips
            # IMPORTANT: Every channel of a state is of size (column_size, row_size), which is the TRANSPOSE of the 
            #            grid that is drawn in the GUI. Thus the HORIZONTAL flip is performed on the TRANSPOSE, which
            #            is actually a VERTICAL flip of the grid drawn in the GUI. Therefore, the actions must be 
            #            transformed according to a VERTICAL flip
            if horizontal_flip_mask.any():
                # Horizontally flip the states
                states[horizontal_flip_mask] = F.hflip(states[horizontal_flip_mask])
                # Horizontally flip the next states
                next_states[horizontal_flip_mask] = F.hflip(next_states[horizontal_flip_mask])
                # VERTICALLY flip the actions
                # NB: We should only swap UP and DOWN, which are both even (0 or 2 resp.)
                #     Thus we can filter for UP and DOWN with ((1 + actions[horizontal_flip_mask & move_mask]) % 2)
                actions[horizontal_flip_mask & move_mask] = (actions[horizontal_flip_mask & move_mask] + (((1 + actions[horizontal_flip_mask & move_mask]) % 2) * 2)) % 4

            # Apply random counterclockwise rotations
            # IMPORTANT: Every channel of a state is of size (column_size, row_size), which is the TRANSPOSE of the 
            #            grid that is drawn in the GUI. Thus the COUNTERCLOCKWISE rotation is performed on the TRANSPOSE, which
            #            is actually a CLOCKWISE rotation of the grid drawn in the GUI. Therefore, the actions must be 
            #            transformed according to a CLOCKWISE rotation
            if rotation_mask.any():
                # Randomly draw the rotation multiplicities (either 1, 2 or 3)
                # NB: For simpliity, we draw a rotation multiplicity for every tuple, even if the rotation_mask
                #     at the respective index is 0
                rotation_multip = torch.randint(1, 4, size=(batch_size, ), device=self.device)
                # Counterclockwise otate the states         
                states = torch.stack([
                    torch.rot90(state, k=multip.item(), dims=[1, 2]) if mask else state
                    for state, mask, multip in zip(states, rotation_mask, rotation_multip)
                ])
                # Counterclockwise rotate the next_states    
                next_states = torch.stack([
                    torch.rot90(next_state, k=multip.item(), dims=[1, 2]) if mask else next_state
                    for next_state, mask, multip in zip(next_states, rotation_mask, rotation_multip)
                ])
                # CLOCKWISE rotate the actions
                actions[rotation_mask & move_mask] = (actions[rotation_mask & move_mask] + rotation_multip[rotation_mask & move_mask]) % 4

            # # Assert that horizontal flips are properly implemented in isolation
            # for i in range(batch_size):
            #     for j in range(state_shape[0]):
            #         for x in range(state_shape[2]):
            #             for y in range(state_shape[1]):
            #                 if horizontal_flip_mask[i]:
            #                     # Again, since the we store the TRANSPOSED of the grid drawn on the GUI,
            #                     # a horizontal flip is actually a vertical flip
            #                     assert states[i][j][x][state_shape[1] - 1 - y].item() == old_states[i][j][x][y].item()
            #                     assert next_states[i][j][x][state_shape[1] - 1 - y].item() == old_next_states[i][j][x][y].item()
            #                     if old_actions[i].item() < 4 and old_actions[i].item() % 2 == 0:
            #                         assert actions[i].item() % 2 == 0 and actions[i].item() == (old_actions[i].item() + 2) % 4
            #                     else: 
            #                         assert actions[i].item() == old_actions[i].item()
            #                 else:
            #                     assert states[i][j][x][y].item() == old_states[i][j][x][y].item()
            #                     assert next_states[i][j][x][y].item() == old_next_states[i][j][x][y].item()
            #                     assert actions[i].item() == old_actions[i].item()

            # # Assert that counterclockwise rotations are properly implemented in isolation
            # for i in range(batch_size):
            #     for j in range(state_shape[0]):
            #         for x in range(state_shape[2]):
            #             for y in range(state_shape[1]):
            #                 if rotation_mask[i]:
            #                     # Again, since the we store the TRANSPOSED of the grid drawn on the GUI,
            #                     # a counterclockwise rotation is actually a clockwise rotation
            #                     if rotation_multip[i] == 1:
            #                         assert states[i][j][state_shape[1] - 1 - y][x].item() == old_states[i][j][x][y].item()
            #                         assert next_states[i][j][state_shape[1] - 1 - y][x].item() == old_next_states[i][j][x][y].item()
            #                     elif rotation_multip[i] == 2:
            #                         assert states[i][j][state_shape[1] - 1 - x][state_shape[2] - 1 - y].item() == old_states[i][j][x][y].item()
            #                         assert next_states[i][j][state_shape[1] - 1 - x][state_shape[2] - 1 - y].item() == old_next_states[i][j][x][y].item()
            #                     elif rotation_multip[i] == 3:
            #                         assert states[i][j][y][state_shape[2] - 1 - x].item() == old_states[i][j][x][y].item()
            #                         assert next_states[i][j][y][state_shape[2] - 1 - x].item() == old_next_states[i][j][x][y].item()


            #                     if old_actions[i].item() < 4:
            #                         assert actions[i].item() == (old_actions[i].item() + rotation_multip[i]) % 4
            #                     else: 
            #                         assert actions[i].item() == old_actions[i].item()
            #                 else:
            #                     assert states[i][j][x][y].item() == old_states[i][j][x][y].item()
            #                     assert next_states[i][j][x][y].item() == old_next_states[i][j][x][y].item()
            #                     assert actions[i].item() == old_actions[i].item()

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