import os
import pickle
import json
import torch
import random
import numpy as np
from utils import action_index_to_str

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Load the configuration
    with open("./config/config.json", "r") as file:
        self.CONFIG = json.load(file)

    # Set the device to be used
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the local Q-network
    self.local_q_network = Network(state_size=self.CONFIG["STATE_SIZE"], action_size=self.CONFIG["ACTION_SIZE"]).to(self.device)

    # TODO: Load the model (+ experience replay buffer) it if it exists
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     weights = np.random.rand(len(ACTIONS))
    #     self.model = weights / weights.sum()
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # Select the action using an epsilon-greedy policy
    if self.train and random.random() <= self.exploration_rate:
        # Return a random action
        return action_index_to_str(random.randrange(self.CONFIG["ACTION_SIZE"]))
    else:
        # Set the local network to evaluation mode
        self.local_q_network.eval()
        # Extract the features from the state and convert it to a tensor
        # NB: Add a (dummy) batch dimension with unsqueeze(0) to obtain the shape (batch_size [= 1], column_size, row_size, channel_size)
        state = torch.from_numpy(state_to_features(game_state)).float().unsqueeze(0).to(self.device)
        # Disable gradient tracking
        with torch.no_grad():
            # Get the Q-values of every action for the state from the local Q-network
            # The shape of action_q_values is (batch_size, action_size)
            action_q_values = self.local_q_network(state=state)
        # Set the local network back to training mode
        self.local_q_network.train()
        # Return the action with the highest Q-value (by taking the argmax along the second dimension)
        # NB: The .item() method returns the value of the tensor as a standard Python number
        return action_index_to_str(torch.argmax(action_q_values, dim=1).item())


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
