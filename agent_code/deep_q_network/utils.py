import os 
import random
import torch
import wandb
import numpy as np
from typing import Tuple, Union
import settings as s

def set_seed(seed: int, change_world_seed, use_cuda: bool = False) -> None:
  """ Set the seed for the random number generators in Python, NumPy and PyTorch.

  Args:
      seed (int): Seed for the random number generators.
      change_world_seed (function): Function to change the seed of the world.
      use_cuda (bool, optional): Whether CUDA is used. Defaults to False.
  """

  # Set the seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  change_world_seed(seed)

  # Set the seed and the deterministic behavior for CUDA
  if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unset_seed(change_world_seed, use_cuda: bool = False) -> None:
    """ Reset the seed settings to allow for non-deterministic behavior.

    Args:
        change_world_seed (function): Function to change the seed of the world.
        use_cuda (bool, optional): Whether CUDA is used. Defaults to False.
    """

    # Reset to default (non-deterministic) settings
    random.seed() 
    np.random.seed()  
    torch.manual_seed(np.random.randint(0, 2**32 - 1))
    change_world_seed(None)

    if use_cuda:
        torch.cuda.manual_seed(np.random.randint(0, 2**32 - 1))
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def action_str_to_index(action: str) -> int:
  """ Convert an action string to an index.

  Args:
      action (str): The action as a string.

  Returns:
      int: The index of the action.
  """

  # Define the mapping from the action strings to the indices
  action_mapping = {
      "UP": 0,
      "RIGHT": 1,
      "DOWN": 2,
      "LEFT": 3,
      "WAIT": 4,
      "BOMB": 5
  }

  # Return the index of the action
  return action_mapping[action]


def action_index_to_str(index: int) -> str:
  """ Convert an action index to a string.

  Args:
      index (int): The index of the action.

  Returns:
      str: The action as a string.
  """

  # Define the mapping from the indices to the action strings
  action_mapping = {
      0: "UP",
      1: "RIGHT",
      2: "DOWN",
      3: "LEFT",
      4: "WAIT",
      5: "BOMB"
  }

  # Return the action as a string
  return action_mapping[index]


def crop_channel(channel: np.ndarray, border_size: int) -> np.ndarray:
    """ Crop the channel by removing the border.

    Args:
        channel (np.array): The channel to crop.
        border_size (int): The size of the border to remove.

    Returns:
        np.array: The cropped channel.
    """

    # Crop the channel by removing the border
    return channel[border_size:-border_size, border_size:-border_size]


def get_bomb_blast_coords(x: int, y: int, arena: np.ndarray) -> list:
    """ Get the coordinates of the blast of a bomb. Taken from item.py.
        NB: The blast only stops at walls but it does not stop at crates, players, coins or other bombs.

    Args:
        x (int): The x-coordinate of the bomb.
        y (int): The y-coordinate of the bomb.
        arena (np.ndarray): The current state of the arena/field.

    Returns:
        list: The coordinates of the blast of the bomb.
    """

    blast_coords = [(x, y)]
    power = s.BOMB_POWER

    for i in range(1, power + 1):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, power + 1):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, power + 1):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, power + 1):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords


def is_bomb_useless(x: int, y: int, arena: np.ndarray) -> bool:
    """ Determine wheter the dropped bomb is useless.

    Args:
        x (int): The x-coordinate of the bomb.
        y (int): The y-coordinate of the bomb.
        arena (np.ndarray): The current state of the arena/field.

    Returns:
        bool: Whether the dropped bomb is useless.
    """

    # Get the blast coordinates of the bomb
    blast_coords = get_bomb_blast_coords(x, y, arena)
    # Check if the bomb hits any crate
    for coord in blast_coords:
        if arena[coord] == 1:
            # If so, the bomb is not useless
            return False
    # Otherwise the bomb is useless
    return True


def round_ended_but_not_dead(self, game_state: dict) -> bool:
    """ Check if the the game/round ended but the agent is not dead.
        In this case, the last step is handled twice (in game_events_occurred and in end_of_round) and thus should be ignored once during training.

    Args:
        game_state (dict): The current game state.

    Returns:
        bool: Whether the game/round ended but the agent is not dead.
    """

    # Check if there are no opponents, no crates, no coins, no bombs and no explosions left
    if (len(game_state["others"]) == 0
            and (game_state["field"] == 1).sum() == 0
            and len(game_state["coins"]) == 0
            and len(game_state["bombs"]) == 0
            and (game_state["explosion_map"] >= 1).sum() == 0):
        return True

    # Check if the maximum number of steps has been reached
    if game_state["step"] >= s.MAX_STEPS:
        return True

    return False


def save_data(project_name: str, run, run_name: str, wandbAPI, metadata: dict, network, optimizer, test_best_avg_score: float, buffer, path: str) -> None:
    """ Save the data.

    Args:
        project_name (str): The name of the project.
        run: The W&B run.
        run_name (str): The name of the run.
        wandbAPI: The W&B API.
        metadata (dict): The metadata to save.
        network (Network): The deep Q-network.
        optimizer: The optimizer of the network.
        buffer (ExperienceReplayBuffer): The experience replay buffer.
        test_best_avg_score (float): The best average score achieved during testing.
        path (str): The path to save the data.
    """

    # Create a dictionary to store the training_data
    training_data = {
        "optimizer": optimizer.state_dict(),
        "exploration_rate": metadata["exploration_rate"],
        "test_best_avg_score": test_best_avg_score,
        "training_steps": metadata["training_steps"],
        "training_rounds": metadata["training_rounds"]
    }

    # Build the path
    # NB: The path is either "best/" or "last/" depending on whether the best average score was achieved during testing
    path = os.path.join(path, "best/" if metadata["is_test_best_avg_score"] else "last/")

    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Define the paths
    network_path = os.path.join(path, "network.pth")
    training_data_path = os.path.join(path, "training_data.pth")
    #buffer_path = os.path.join(path, "experience_replay_buffer.pkl")

    # Save the network to a file
    with open(network_path, 'wb') as f:
        torch.save(network.state_dict(), f)
    # Save the training_data to a file
    with open(training_data_path, 'wb') as f:
        torch.save(training_data, f)
    # Save the buffer to a file
    #buffer.save(buffer_path)

    # Create a new artifact
    # NB: The name of the artifact is either "best" or "last" depending on whether the best average score was achieved during testing
    artifact_name = f"{run_name}_{"best" if metadata["is_test_best_avg_score"] else "last"}"
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(network_path)
    artifact.add_file(training_data_path)
    #artifact.add_file(buffer_path)

    # Log the artifact to W&B
    run.log_artifact(artifact)
    try:
        # Get all versions of the artifact
        artifact_versions = wandbAPI.artifacts("model", f"{project_name}/{artifact_name}")
        # Keep 3 best versions and the last version
        keep_number = 3 if metadata["is_test_best_avg_score"] else 1
        
        # Check if there are versions to delete
        if len(artifact_versions) > keep_number:
            if metadata["is_test_best_avg_score"]:
                # Sort artifact versions by test_avg_score, highest first
                artifact_versions = sorted(artifact_versions, key=lambda art: art.metadata["test_avg_score"], reverse=True)
            else:
                # Sort artifact versions by creation time, latest first
                artifact_versions = sorted(artifact_versions, key=lambda art: art.created_at, reverse=True)

            # Delete old/bad artifact versions
            for art in artifact_versions[keep_number:]:
                art.delete()
    except Exception as e:
        print("Error in deleting old artifact versions")
        print(e)


def load_network(network, path: str, device: torch.device) -> None:
    """ Load the network.

    Args:
        network (Network): The network to load the state_dict into.
        path (str): The path to the network.
        device (torch.device): The device to load the network on.
    """
    network_path = os.path.join(path, "network.pth")
    # Load the network
    with open(network_path, 'rb') as f:
        network.load_state_dict(torch.load(f, map_location=device, weights_only=True))


def load_training_data(optimizer, buffer, path: str, device: torch.device) -> Tuple[float, float, int, int]:
    """ Load the training data and the buffer.

    Args:
        optimizer: The optimizer of the network.
        buffer (ExperienceReplayBuffer): The experience replay buffer.
        path (str): The path to the training data and the buffer.
        device (torch.device): The device to load the data on

    Returns:
        Tuple[float, float, int, int]: The exploration rate, the best average score achieved during testing, the number of training steps and the number of training rounds.
    """

    # Define the paths
    training_data_path = os.path.join(path, "training_data.pth")
    #buffer_path = os.path.join(path, "experience_replay_buffer.pkl")

    training_data = None
    # Load the training data
    with open(training_data_path, 'rb') as f:
        training_data = torch.load(f, map_location=device, weights_only=True)

    optimizer.load_state_dict(training_data['optimizer'])

    # Move the optimizer to the device
    move_optimizer_to_device(optimizer, device)

    exploration_rate = training_data['exploration_rate']
    test_best_avg_score = training_data['test_best_avg_score']
    training_steps = training_data['training_steps']
    training_rounds = training_data['training_rounds']

    # Load the buffer
    #buffer.load(buffer_path)
    
    return exploration_rate, test_best_avg_score, training_steps, training_rounds


def move_optimizer_to_device(optimizer, device: torch.device) -> None:
    """ Move the optimizer to the device, since this is not done automatically.
        See: https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385

    Args:
        optimizer: The optimizer to move.
        device (torch.device): The device to move the optimizer to.
    """
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def distance(a: Tuple, b: Tuple) -> int:
    """ Calculate the (Manhattan) distance between two points a and b in the grid.
    Args:
        a (Tuple): The first point.
        b (Tuple): The second point.

    Returns:
        int: The (Manhattan) distance between the two points.
    """

    # Check if the points are the same
    if a == b:
        return 0

    # Calculate the Manhattan distance between the two points
    distance = abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Note that two points with the same even x coordinate OR the same even y coordinate have an
    # additional distance of 2 between them because of the walls between them, which must be circumvented.
    if a[0] == b[0] and a[0] % 2 == 0 or a[1] == b[1] and a[1] % 2 == 0:
        distance += 2

    return distance


def potential_of_state(state: Union[dict, None]) -> float:
    """ Calculate the potential of the state.

    Args:
        state (Union[dict, None]): The state to calculate the potential of.

    Returns:
        float: The potential of the state.
    """

    # Check if the state is None
    if state is None:
        return 0.0

    # Get the position of the agent
    agent_position = state["self"][3]

    nearest_coin_distance = None
    # Loop over all coins
    for coin in state["coins"]:
        # Calculate the distance to the coin
        distance_to_coin = distance(agent_position, coin)
        # Update the nearest coin distance
        # NB: Do not consider coins at the agent's position (which can happen in the initial state of the coin heaven scenario)
        if (nearest_coin_distance == None or distance_to_coin < nearest_coin_distance) and distance_to_coin != 0:
            nearest_coin_distance = distance_to_coin

    if nearest_coin_distance is None:
        return 0.0

    return 1.2 ** (-nearest_coin_distance)