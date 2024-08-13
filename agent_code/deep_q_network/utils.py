import random
import torch
import numpy as np

def set_seed(seed: int, use_cuda: bool = False) -> None:
  """ Set the seed for the random number generators in Python, NumPy and PyTorch.

  Args:
      seed (int): Seed for the random number generators.
      use_cuda (bool, optional): Whether CUDA is used. Defaults to False.
  """

  # Set the seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set the seed and the deterministic behavior for CUDA
  if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

