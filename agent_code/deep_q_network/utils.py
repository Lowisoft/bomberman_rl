import random
import torch
import numpy as np
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


def get_bomb_blast_coords(x: int, y: int, power: int, arena: np.ndarray) -> list:
    """ Get the coordinates of the blast of a bomb. Taken from item.py.
        NB: The blast only stops at walls but it does not stop at crates, players, coins or other bombs.

    Args:
        x (int): The x-coordinate of the bomb.
        y (int): The y-coordinate of the bomb.
        power (int): The power of the bomb.
        arena (np.ndarray): The current state of the arena/field.

    Returns:
        list: The coordinates of the blast of the bomb.
    """

    blast_coords = [(x, y)]

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