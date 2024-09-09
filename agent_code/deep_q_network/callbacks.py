import os
import yaml
import torch
import random
import numpy as np
import copy # ONLY FOR TRAINING
from collections import deque # ONLY FOR TRAINING
import wandb # ONLY FOR TRAINING
from datetime import datetime # ONLY FOR TRAINING
import settings as s
from typing import Union
from .utils import (
    action_index_to_str, 
    crop_channel, 
    get_bomb_blast_coords, 
    load_network, 
    action_str_to_index,
    num_crates_and_opponents_in_blast_coords,
    distance_to_best_coin,
    distance_to_nearest_crate
)
from .model.network import Network
#from .model.coin_collector_agent import coin_collector_act

def setup(self) -> None:
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
    with open("./config/base_config.yaml", "r") as file:
        self.CONFIG = yaml.safe_load(file)

    if self.train:
        # Get the current time
        now = datetime.now()
        
        # Create a unique name for the run
        self.run_name = f"{self.CONFIG['PROJECT_NAME_SHORT']}_{now.strftime("%y%m%d%H%M%S")}"

        # Initialize the wandb run
        self.run = wandb.init(
            project=self.CONFIG["PROJECT_NAME"],  
            name=self.run_name,
            config=self.CONFIG # Gets automatically overwritten by config of the sweep (if available)
        )
        # Make sure the same configuration is used for the agent and the wandb run
        self.CONFIG = self.run.config

        # Check if we should break loops
        if self.CONFIG["BREAK_LOOPS"]:
            # If so, initialize the loop buffer to detect and break loops
            self.loop_buffer = deque(maxlen=self.CONFIG["LOOP_BUFFER_CAPACITY"])

    # Set the device to be used
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    print("Device:", self.device)

    # Initialize the local Q-network
    self.local_q_network = Network(
        channel_size=self.CONFIG["CHANNEL_SIZE"], 
        column_size=(s.COLS - 2), 
        row_size=(s.ROWS - 2), 
        action_size=self.CONFIG["ACTION_SIZE"], 
        hidden_layer_size=self.CONFIG["HIDDEN_LAYER_SIZE"],
        use_dueling_dqn=self.CONFIG["USE_DUELING_DQN"]
        ).to(self.device)

    # Check if we can start from a saved state
    if "START_FROM" in self.CONFIG and self.CONFIG["START_FROM"] in ["best", "last"] and os.path.exists(f"{self.CONFIG["PATH"]}/{self.CONFIG["START_FROM"]}/"):
        print(f"Loading {self.CONFIG["START_FROM"]} network from saved state.")
        load_network(network=self.local_q_network, path=f"{self.CONFIG["PATH"]}/{self.CONFIG["START_FROM"]}/", device=self.device)
    # Otherwise, set up the model from scratch
    else:
        print("Setting up network from scratch.")
        # Initialize the weights of the local Q-network using the Kaiming initialization
        self.local_q_network.initialize_weights_kaiming()

    # Set the local network to evaluation mode
    self.local_q_network.eval()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # Initialize the action
    action = "WAIT"
    # Get the features from the state
    features = state_to_features(game_state)

    # Select the action using an epsilon-greedy policy
    # NB: If the agent is not trained or if the agent is tested during training, the exploration is disabled
    if self.train and not self.test_training and random.random() <= self.exploration_rate:
        # Return a random action
        action = action_index_to_str(random.randrange(self.CONFIG["ACTION_SIZE"]))
    else:
        # Extract the features from the state and convert it to a tensor
        # NB: Add a (dummy) batch dimension with unsqueeze(0) to obtain the shape (batch_size [= 1], channel_size, column_size, row_size)
        state = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        # Disable gradient tracking
        with torch.no_grad():
            # Get the Q-values of every action for the state from the local Q-network
            # The shape of action_q_values is (batch_size, action_size)
            action_q_values = self.local_q_network(state)
        # Return the action with the highest Q-value (by taking the argmax along the second dimension)
        # NB: The .item() method returns the value of the tensor as a standard Python number
        action = action_index_to_str(torch.argmax(action_q_values, dim=1).item())

    # Break loops (if enabled) by avoiding repeating state-action pairs
    # NB: If the agent is not trained or if the agent is tested during training, loops are not broken
    if self.CONFIG["BREAK_LOOPS"] and self.train and not self.test_training:   
        # Hash the features so that we can compare it better with recent states/features
        features_hash = hash(features.tobytes())
        # Count the number of repeating state-action pairs
        repetitions = 0
        repetition_indices = []
        # Loop over the recent state-action pairs in the reversed order
        for rev_index, recent_state_action in enumerate(reversed(self.loop_buffer)):
            # Check for a repetition
            if recent_state_action[0] == action and recent_state_action[1] == features_hash and np.array_equal(recent_state_action[2], features):
                repetitions += 1
                # Get the forward index
                index = len(self.loop_buffer) - 1 - rev_index
                repetition_indices.append(index)
            # If we have found at least 2 repetitions (so 3 equal state-action pairs in total), then break the loop
            # NB: We also check if there is no bomb dropped
            if repetitions >= 2 and game_state["self"][2]:
                # Store the old action
                old_action = action
                # Initialize the forbidden actions
                forbidden_actions = [old_action, "WAIT"]
                # Get the number of crates that would be attacked if the agent would place a bomb
                num_crates_attacked, num_opponents_attacked = num_crates_and_opponents_in_blast_coords(np.array(game_state["self"][3]), game_state["field"], game_state["others"], s.BOMB_POWER)
                # If BOMB would result in a useless bomb, forbid it
                if num_crates_attacked == 0 and num_opponents_attacked == 0:
                    forbidden_actions.append("BOMB")
                # Get the current position of the agent
                curr_pos = game_state["self"][3]
                # Add invalid moves to the forbidden actions
                if (game_state["field"][curr_pos[0]][curr_pos[1] - 1] != 0):
                    forbidden_actions.append("UP")
                if (game_state["field"][curr_pos[0]][curr_pos[1] + 1] != 0):
                    forbidden_actions.append("DOWN")
                if (game_state["field"][curr_pos[0] - 1][curr_pos[1]] != 0):
                    forbidden_actions.append("LEFT")
                if (game_state["field"][curr_pos[0] + 1][curr_pos[1]] != 0):
                    forbidden_actions.append("RIGHT")
                
                # Check if the repetition is a 1-loop, i.e. WAIT or INVALID_ACTION
                if repetition_indices[0] == repetition_indices[1] + 1 and len(self.loop_buffer) == repetition_indices[0] + 1:
                    # Perform a random action but exclude the forbidden actions
                    while action in forbidden_actions:
                        action = action_index_to_str(random.randrange(self.CONFIG["ACTION_SIZE"]))
                # Otherwise check if the repetition is a 2-loop with a moving action, i.e. LEFT & RIGHT or UP & DOWN
                elif repetition_indices[0] == repetition_indices[1] + 2 and len(self.loop_buffer) == repetition_indices[0] + 2 and action_str_to_index(action) < 4:
                    # Copy the next game state
                    next_game_state = game_state.copy()
                    # Compute the next position
                    next_pos = [game_state["self"][3][0], game_state["self"][3][1]]
                    if action == 'UP':
                        next_pos[1] -= 1
                    elif action == 'DOWN':
                        next_pos[1] += 1
                    elif action == 'LEFT':
                        next_pos[0] -= 1
                    elif action == 'RIGHT':
                        next_pos[0] += 1
                    # Update the next game state
                    next_game_state["self"] = list(next_game_state["self"])
                    next_game_state["self"][3] = tuple(next_pos)
                    next_game_state["self"] = tuple(next_game_state["self"])
                    # Check if there is at least one revealed coin
                    if len(game_state["coins"]) > 0:
                        # If so, check if the chosen action is bad, i.e. does not reduce the distance to the best coin
                        next_distance = distance_to_best_coin(next_game_state)
                        curr_distance = distance_to_best_coin(game_state)
                        if next_distance is not None and curr_distance is not None and next_distance >= curr_distance:
                             # Perform a random action but exclude the forbidden actions
                            while action in forbidden_actions:
                                action = action_index_to_str(random.randrange(self.CONFIG["ACTION_SIZE"]))
                    else:
                        # Otherwise, check if the chosen action is bad, i.e. does not reduce the distance to the nearest crate
                        next_distance = distance_to_nearest_crate(next_game_state)
                        curr_distance = distance_to_nearest_crate(game_state)
                        if next_distance is not None and curr_distance is not None and next_distance >= curr_distance:
                            # Perform a random action but exclude the forbidden actions
                            while action in forbidden_actions:
                                action = action_index_to_str(random.randrange(self.CONFIG["ACTION_SIZE"]))

                # Increment the number of broken loops in the round/game during training
                if action is not old_action:
                    self.training_broken_loops_of_round += 1
                break
        # Append the current state-action pair to the loop buffer
        self.loop_buffer.append((action, features_hash, features))
        
    #print("Action:", action)
    return game_state['user_input']


def state_to_features(game_state: Union[dict, None]) -> np.array:
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

    # NB: Define the border_size to crop the game board by excluding the walls at the borders
    border_size = 1

    # Initialize the channels
    channels = []

    # Create a channel for the player
    player_channel = np.zeros((s.COLS, s.ROWS))
    # Set the value of the player to 1
    player_channel[game_state["self"][3]] = 1
    # Add the channel to the list of channels
    channels.append(crop_channel(player_channel, border_size))
    
    # Create a channel for the coins
    coins_channel = np.zeros((s.COLS, s.ROWS))
    # Loop over all coins
    for coin in game_state["coins"]:
        # Set the values of the coins to 1 at the respective coordinates
        coins_channel[coin] = 1
    # Add the channel to the list of channels
    channels.append(crop_channel(coins_channel, border_size))

    # Create a channel for the bombs and explosions
    # NB: The values of the bombs channel range from 0 to 5, where 1-4 represent the danger of the bomb and 5 represents the explosion
    # NB: It is important to create a copy of the explosion map to avoid changing the original field
    bombs_channel = game_state["explosion_map"].copy() * (s.BOMB_TIMER + 1)
    # Loop over all bombs
    for bomb in game_state["bombs"]:
        # Calculate the danger of the bomb (ranges from 1 to 4)
        danger = (s.BOMB_TIMER) - bomb[1]
        # Get the coordinates of the bomb blast
        blast_coords = get_bomb_blast_coords(bomb[0][0], bomb[0][1], game_state["field"], s.BOMB_POWER)
        # Loop over all coordinates of the bomb blast
        for coord in blast_coords:
            # Set the value of the bomb blast to the danger of the bomb if there is not already a stronger danger
            if bombs_channel[coord] < danger:
                bombs_channel[coord] = danger
    # Normalize the values of the bombs channel to be in the range [0, 1]
    bombs_channel /= (s.BOMB_TIMER + 1)
    # Add the channel to the list of channels
    channels.append(crop_channel(bombs_channel, border_size))

    # Create a channel for the walls
    # NB: It is important to create a copy of the field to avoid changing the original field
    walls_channel = game_state["field"].copy()
    # Remove the crates from the field
    walls_channel[walls_channel == 1] = 0.
    # Set the values of the walls to 1
    walls_channel[walls_channel == -1] = 1.
    # Add the channel to the list of channels
    channels.append(crop_channel(walls_channel, border_size))

    # Create a channel for the crates
    # NB: It is important to create a copy of the field to avoid changing the original field
    crates_channel = game_state["field"].copy()
    # Remove the walls from the field
    crates_channel[crates_channel == -1] = 0.
    # Add the channel to the list of channels
    channels.append(crop_channel(crates_channel, border_size))

    # Create a channel for the opponents
    opponents_channel = np.zeros((s.COLS, s.ROWS))
    # Loop over all opponents
    for opponent in game_state["others"]:
        # Set the value of the opponent to 1
        opponents_channel[opponent[3]] = 1
    # Add the channel to the list of channels
    channels.append(crop_channel(opponents_channel, border_size))

    # Stack the channels to obtain the shape (channel_size, column_size, row_size)
    stacked_channels = np.stack(channels)

    return stacked_channels
