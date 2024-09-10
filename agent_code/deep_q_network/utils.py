import os 
import random
import torch
import wandb
import numpy as np
from collections import deque
from pathfinding.core.grid import Grid
from pathfinding.finder.bi_a_star import BiAStarFinder
from typing import Tuple, Union, List
import settings as s
import events as e

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


def end_reason_str_to_index(end_reason: str) -> int:
  """ Convert an end reason string to an index.

  Args:
      end_reason (str): The end reason as a string.

  Returns:
      int: The index of the end reasion.
  """

  # Define the mapping from the end reason strings to the indices
  end_reason_mapping = {
      e.KILLED_SELF: 0,
      e.GOT_KILLED: 1,
      e.SURVIVED_ROUND: 2
  }

  # Return the index of the end reason
  return end_reason_mapping[end_reason]


def crop_channel(channel: np.ndarray, border_size: int) -> np.ndarray:
    """ Crop the channel by removing the border.

    Args:
        channel (np.ndarray): The channel to crop.
        border_size (int): The size of the border to remove.

    Returns:
        np.ndarray: The cropped channel.
    """

    # Crop the channel by removing the border
    return channel[border_size:-border_size, border_size:-border_size]


def get_bomb_blast_coords(x: int, y: int, field: np.ndarray, power: int = s.BOMB_POWER) -> list:
    """ Get the coordinates of the blast of a bomb. Taken from item.py.
        NB: The blast only stops at walls but it does not stop at crates, players, coins or other bombs.

    Args:
        x (int): The x-coordinate of the bomb.
        y (int): The y-coordinate of the bomb.
        field (np.ndarray): The current field of the game.
        power (int, optional): The power of the bomb. Defaults to s.BOMB_POWER.

    Returns:
        list: The coordinates of the blast of the bomb.
    """

    blast_coords = [(x, y)]

    for i in range(1, power + 1):
        if field[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, power + 1):
        if field[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, power + 1):
        if field[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, power + 1):
        if field[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords


def num_crates_and_opponents_in_blast_coords(pos: np.ndarray, field: np.ndarray, others: List) -> Tuple[int, int]:
    """ Return the number of crates and the number of oppponents that are in the blast coords of the given bomb position.

    Args:
        pos (np.ndarray): The postion of the bomb.
        field (np.ndarray): The current field of the game.
        others (List): The current list of opponents.

    Returns:
        Tuple[int, int]: The number of crates and the number of opponents that are in the blast coords of the given bomb position.
    """

    # Initialize the number of crates
    num_crates = 0
    # Initialize the number of opponents
    num_opponents = 0
    # Get the blast coordinates of the bomb
    blast_coords = get_bomb_blast_coords(pos[0], pos[1], field, s.BOMB_POWER)
    for coord in blast_coords:
        # Check if there is a crate at the coord
        if field[coord] == 1:
            # If so, increase the number of crates
            num_crates += 1
    # Get the number of opponents in the blast coordinates
    num_opponents = sum(1 for other in others if other[3] in blast_coords)

    return num_crates, num_opponents
    

def is_bomb_at_pos(pos: np.ndarray, bombs: List) -> bool:
    """ Checks whether there is a bomb at the given position.

    Args:
        pos (np.ndarray): Position to check for.
        bombs (List): The current list of bombs.

    Returns:
        bool: Whether there is a bomb at the given position.
    """

    return any(np.array_equal(np.array(bomb[0]), pos) for bomb in bombs)


def is_opponent_at_pos(pos: np.ndarray, others: List) -> bool:
    """ Checks whether there is an opponent at the given position.

    Args:
        pos (np.ndarray): Position to check for.
        others (List): The current list of opponents.

    Returns:
        bool: Whether there is an opponent at the given position.
    """

    return any(np.array_equal(np.array(other[3]), pos) for other in others)


def is_crate_or_wall_at_pos(pos: np.ndarray, field: np.ndarray) -> bool:
    """ Checks whether there is a crate or a wall at the given position.

    Args:
        pos (np.ndarray): Position to check for
        field (np.ndarray): The current field of the game.

    Returns:
        bool: Whether there is a crate or a wall at the given position.
    """

    return field[tuple(pos)] != 0


def is_pos_blocked(pos: np.ndarray, state: dict) -> bool:
    """ Returns whether the given position is blocked.
    Args:
        pos (np.ndarray): Position to check for.
        state (dict): The current state of the game.

    Returns:
        bool: Whether whether the given position is blocked.
    """

    return is_bomb_at_pos(pos, state["bombs"]) or is_opponent_at_pos(pos, state["others"]) or is_crate_or_wall_at_pos(pos, state["field"])


def agent_has_trapped_itself(state: dict,  action: str, next_state: Union[dict, None], events: List[str]) -> bool:
    """ Returns whether the agent has trapped itself.
        TODO: Maybe consider whether crates will be destroyed by other bombs and will thus un-trap the agent

    Args:
        state (dict): The state before the action was taken.
        action (str): The action taken in the state.
        next_state (Union[dict, None]): The state after the action was taken. None if the round/game.
        events (List[str]): The events that occurred when going from the state to the next state.

    Returns:
        bool: Whether the agent has trapped istelf
    """

    # For simplicity, do not consider the last action of a round/game
    if next_state is None:
        return False

    can_place_bomb = state["self"][2]
    curr_pos = np.array(state["self"][3])
    next_pos = np.array(next_state["self"][3])

    # Check if the agent stood on its own bomb and moved away from it (either UP, RIGHT, DOWN or LEFT)
    # NB: The agent can only stand on its own bomb, not on the bomb of opponents
    if (not can_place_bomb 
        and is_bomb_at_pos(curr_pos, state["bombs"]) 
        and action_str_to_index(action) < 4
        # Make sure that the action was really performed (e.g. no invalid action)
        and any(moving_event in events for moving_event in [e.MOVED_UP, e.MOVED_RIGHT, e.MOVED_DOWN, e.MOVED_LEFT])):
        # Get the (normalized) moving direction of the agent
        moving_direction = next_pos - curr_pos
        # Get the (normalized) direction that is perpendicular to the moving direction
        perpendicular_direction = (1, 1) - abs(moving_direction)
        # Initialize the booleans
        can_escape_in_straight_line = True
        can_escape_around_corner = False
        # Loop over the power of the bomb to explore the straight line away from the bomb and its two parallel neighboring rows/columns
        for i in range(1, s.BOMB_POWER + 1):    
            # Check if it is possible to escape around the corner, i.e. go to a different row AND column than the one of the bomb
            # For this, we use the perpendicular direction to get the positions on the two rows/columns that are PARALLEL to the 
            # straight line away from the bomb
            # IMPORTANT: We can only escape around the corner if can_escape_in_straight_line is still true for the current iteration,
            #            because we must be able to follow the straight line far enough to be able to actually go around the corner.
            next_corner_1_pos = tuple(next_pos + (i - 1) * moving_direction - perpendicular_direction)
            next_corner_2_pos = tuple(next_pos + (i - 1) * moving_direction + perpendicular_direction)
            if (can_escape_in_straight_line and not can_escape_around_corner
                and (not is_pos_blocked(next_corner_1_pos, next_state)
                    or not is_pos_blocked(next_corner_2_pos, next_state))):
                can_escape_around_corner = True

            # Check if there is any obstacle that blocks escaping in straight line from the bomb
            # NB: The field outside of the explosion radius must also be free, thus i goes up to 3
            next_straight_pos = tuple(next_pos + i * moving_direction)
            if can_escape_in_straight_line and is_pos_blocked(next_straight_pos, next_state):
                can_escape_in_straight_line = False  

        return not can_escape_in_straight_line and not can_escape_around_corner

    return False 


def agents_waits_uselessly(state: dict, action: str, next_state: Union[dict, None], events: List[str]) -> bool:
    """ Returns whether the agent waits uselessly.
        A wait is useless if the agent waits on a blast coordinate OR if the agent waits but no blast coordinate is reacheable by the next step.

    Args:
        state (dict): The state before the action was taken.
        action (str): The action taken in the state.
        next_state (Union[dict, None]): The state after the action was taken. None if the round/game.
        events (List[str]): The events that occurred when going from the state to the next state.

    Returns:    
        bool: Whether the agent waits uselessly.
    """

    if action == "WAIT" and e.WAITED in events:
        agent_pos = state["self"][3]
        # Get the neighboring coords of the agent (excluding the agent's current position, walls and crates)
        neighboring_coords = []
        x = agent_pos[0]
        y = agent_pos[1]
        if state["field"][x + 1, y] == 0:
            neighboring_coords.append((x + 1, y))
        if state["field"][x - 1, y] == 0:
            neighboring_coords.append((x - 1, y))
        if state["field"][x, y + 1] == 0:
            neighboring_coords.append((x, y + 1))
        if state["field"][x, y - 1] == 0:
            neighboring_coords.append((x, y - 1))

        # Get a list of all blast coordinates of all bombs
        all_bomb_blast_coord = []
        for bomb in state["bombs"]:
            all_bomb_blast_coord.extend(get_bomb_blast_coords(bomb[0][0], bomb[0][1], state["field"], s.BOMB_POWER))

        # Check if wait was useless,
        # i.e. it waits on a blast coordinate OR it waits but no blast coordinate is reacheable by the next step
        return agent_pos in all_bomb_blast_coord or all(neighboring_coord not in all_bomb_blast_coord and state["explosion_map"][neighboring_coord] == 0 for neighboring_coord in neighboring_coords) 
    return False


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


def save_data(project_name: str, run, run_name: str, wandbAPI, metadata: dict, network, optimizer, test_best_avg_score: float, buffer, path: str, use_per: bool = False) -> None:
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
        use_per (bool, optional): Whether PER is used. Defaults to False.
    """

    # Create a dictionary to store the training_data
    training_data = {
        "optimizer": optimizer.state_dict(),
        "exploration_rate": metadata["exploration_rate"],
        "test_best_avg_score": test_best_avg_score,
        "training_steps": metadata["training_steps"],
        "training_rounds": metadata["training_rounds"]
    }

    # Add the weight importance to the training_data if PER is used
    if use_per:
        training_data["weight_importance"] = metadata["weight_importance"]

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


def load_training_data(optimizer, buffer, path: str, device: torch.device, use_per: bool = False) -> Tuple[float, Union[float, None], float, int, int]:
    """ Load the training data and the buffer.

    Args:
        optimizer: The optimizer of the network.
        buffer (ExperienceReplayBuffer): The experience replay buffer.
        path (str): The path to the training data and the buffer.
        device (torch.device): The device to load the data on.
        use_per (bool, optional): Whether PER is used. Defaults to False.

    Returns:
        Tuple[float, Union[float, None], float, int, int]: 
            The exploration rate, the weight importance (ONLY USED FOR PRIORITIZED EXPERIENCE REPLAY),
            the best average score achieved during testing, the number of training steps and the number of training rounds.
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
    weight_importance = training_data.get('weight_importance', None) if use_per else None
    test_best_avg_score = training_data['test_best_avg_score']
    training_steps = training_data['training_steps']
    training_rounds = training_data['training_rounds']

    # Load the buffer
    #buffer.load(buffer_path)
    
    return exploration_rate, weight_importance, test_best_avg_score, training_steps, training_rounds


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


def manhattan_distance(a: Tuple, b: Tuple) -> int:
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


def distance_to_best_coin(state: Union[dict, None]) -> Union[float, None]:
    """ Calculate the distance to the best coin (i.e. nearest coin where no other opponent is nearer).
        Returns None if there is no best coin (not visible, not reacheable or other opponents nearer).
    Args:
        state (Union[dict, None]): The state for which compute the distance.

    Returns:
        Union[float, None]: The distance to the best coin.
    """

    # Prepare the grid for the pathfinding algorithm (> 0: obstacle; <= 0: free)
    numpy_grid = np.abs(np.copy(state["field"]))
    # Consider also bombs as obstacles
    for bomb in state["bombs"]:
        numpy_grid[bomb[0]] = 1
    # Consider also opponents as obstacles
    for other in state["others"]:
        numpy_grid[other[3]] = 1
    # Consider also the agent as an obstacle (for other opponents)
    numpy_grid[state["self"][3]] = 1
    # Create the grid for the pathfinding algorithm
    # IMPORTANT: We must transpose the numpy matrix
    grid = Grid(matrix=numpy_grid.T, inverse=True)
    # Initialize the finder of the pathfinding algorithm
    finder = BiAStarFinder()

    # Get the position of the agent
    agent_position = state["self"][3]
    agent_node = grid.node(*agent_position)

    # Find the reachable opponents that could potentially steal the coin
    reachable_opponents = []
    # Loop over all opponents
    for opponent in state["others"]:
        opponent_node = grid.node(*opponent[3])
        path, _ = finder.find_path(agent_node, opponent_node, grid)
        # Calculate the distance to the opponent
        distance_to_opponent = len(path) - 1
        # Check if the opponent is reacheable
        if distance_to_opponent > 0:
            reachable_opponents.append(opponent)
        # Cleanup the grid for the next potential iteration
        grid.cleanup()

    best_coin_distance = None
    # Loop over all coins
    for coin in state["coins"]:
        coin_node = grid.node(*coin)
        path, _ = finder.find_path(agent_node, coin_node, grid)
        # Calculate the distance to the coin
        distance_to_coin = len(path) - 1
        # Cleanup the grid for the next potential iteration
        grid.cleanup()
        # Check if the coin cannot be stolen by any opponent
        for opponent in reachable_opponents:
            opponent_node = grid.node(*opponent[3])
            opponent_path, _ = finder.find_path(opponent_node, coin_node, grid)
            # Calculate the distance to the coin for the opponent
            opponent_distance_to_coin = len(opponent_path) - 1
            # Cleanup the grid for the next potential iteration
            grid.cleanup()
            # Check if the opponent can steal the coin
            if opponent_distance_to_coin > 0 and opponent_distance_to_coin < distance_to_coin:
                # If so, do not consider the coin
                distance_to_coin = -1
        # Update the best coin distance
        # NB: Do not consider coins at the agent's position (which can happen in the initial state of the coin heaven scenario)
        # NB2: Filter out unreachable coins (i.e. distance of -1)
        if (best_coin_distance == None or distance_to_coin < best_coin_distance) and distance_to_coin > 0:
            best_coin_distance = distance_to_coin

    return best_coin_distance


def distance_to_nearest_opponent(state: Union[dict, None]) -> Union[float, None]:
    """ Calculate the distance to the nearest opponent.
        Returns None if there is no (reachable) opponent.
    Args:
        state (Union[dict, None]): The state for which compute the distance.

    Returns:
        Union[float, None]: The distance to the nearest opponent.
    """

    # Prepare the grid for the pathfinding algorithm (> 0: obstacle; <= 0: free)
    numpy_grid = np.abs(np.copy(state["field"]))
    # Consider also bombs as obstacles
    # IMPORTANT: We do not have to consider opponents as obstacles, since we want to find the nearest opponent
    for bomb in state["bombs"]:
        numpy_grid[bomb[0]] = 1
    # Create the grid for the pathfinding algorithm
    # IMPORTANT: We must transpose the numpy matrix
    grid = Grid(matrix=numpy_grid.T, inverse=True)
    # Initialize the finder of the pathfinding algorithm
    finder = BiAStarFinder()

    # Get the position of the agent
    agent_position = state["self"][3]
    agent_node = grid.node(*agent_position)

    nearest_opponent_distance = None
    # Loop over all opponents
    for opponent in state["others"]:
        opponent_node = grid.node(*opponent[3])
        path, _ = finder.find_path(agent_node, opponent_node, grid)
        # Calculate the distance to the opponent
        distance_to_opponent = len(path) - 1
        # Cleanup the grid for the next potential iteration
        grid.cleanup()
        # Update the best opponent distance
        # NB: Filter out unreachable opponents (i.e. distance of -1)
        if (nearest_opponent_distance == None or distance_to_opponent < nearest_opponent_distance) and distance_to_opponent > 0:
            nearest_opponent_distance = distance_to_opponent

    return nearest_opponent_distance


def distance_to_nearest_crate(state: Union[dict, None]) -> Union[float, None]:
    """ Calculate the distance to the nearest coin.
        Returns None if there is no visible crate.
    Args:
        state (Union[dict, None]): The state for which compute the distance.

    Returns:
        Union[float, None]: The distance to the nearest crate.
    """

    # Prepare the grid for the pathfinding algorithm (-1: obstacle; 0: free, 1: crate)
    field = np.copy(state["field"])
    # Consider also bombs as obstacles
    # IMPORTANT: We do not have to consider opponents as obstacles, since if there is an opponent blocking the nearest crate, 
    #            the agent will target the opponent instead of the crate in potential_of_state
    for bomb in state["bombs"]:
        field[bomb[0]] = -1
    
    # Get the position of the agent
    agent_position = state["self"][3]

    # Get a list of all blast coordinates of all bombs
    all_bomb_blast_coord = []
    for bomb in state["bombs"]:
        all_bomb_blast_coord.extend(get_bomb_blast_coords(bomb[0][0], bomb[0][1], state["field"], s.BOMB_POWER))

    # Directions for movement: UP, RIGHT, LEFT, DOWN
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize the BFS queue and visited set
    queue = deque()
    queue.append([agent_position, 0])
    visited = set()
    visited.add(agent_position)

    # Perform BFS
    while queue:
        (curr_x, curr_y), distance = queue.popleft()
        # Check if the current position is a crate
        if field[curr_x][curr_y] == 1:
            # If so, we have found the distance to the nearest crate
            return distance 

        # Otherwise, explore neighbors
        for dx, dy in directions:
            neighbor_x, neighbor_y = curr_x + dx, curr_y + dy

            # Check if the neighbor is within the grid bounds
            if 0 <= neighbor_x < s.COLS and 0 <= neighbor_y < s.ROWS:
                # Check if the neighbor is not an obstacle and not visited
                # IMPORTANT: If the neighbor is a crate, we must ensure that neither the current position (i.e. last position of path to crate)
                #            nor the position of the crate itself is in any of the bomb blast coordinates. 
                #            This ensures that no sub-optimal crate is targeted.
                if ((neighbor_x, neighbor_y) not in visited 
                    and (field[neighbor_x][neighbor_y] == 0 # Neighbor is a free tile
                        or (field[neighbor_x][neighbor_y] == 1 # Neighbor is a crate
                           and not (curr_x, curr_y) in all_bomb_blast_coord
                           and state["explosion_map"][curr_x][curr_y] == 0
                           and not (neighbor_x, neighbor_y) in all_bomb_blast_coord
                           and state["explosion_map"][neighbor_x][neighbor_y] == 0))):
                    visited.add((neighbor_x, neighbor_y))
                    queue.append([(neighbor_x, neighbor_y), distance + 1])

    # When the queue is empty, we have not found a nearest create
    return None


def potential_of_state(state: Union[dict, None], add_state: Union[np.ndarray, None]) -> float:
    """ Calculate the potential of the state.

    Args:
        state (Union[dict, None]): The state to calculate the potential of.
        add_state (Union[np.ndarray, None]): Additional state information to calculate the potential of.

    Returns:
        float: The potential of the state.
    """

    # Check if the state is None
    if state is None:
        return 0.0

    # Get the distance to the best coin
    best_coin_distance = distance_to_best_coin(state) if len(state["coins"]) > 0 else None

    # Check if there is any coin to target
    if best_coin_distance is not None:
        # Return the potential based on the distance to the best coin
        return 1.2 ** (-best_coin_distance) if best_coin_distance is not None else 0.0
    else:
        # Otherwise, get the distance to the nearest crate and to the nearest opponent
        nearest_crate_distance = distance_to_nearest_crate(state)
        nearest_opponent_distance = distance_to_nearest_opponent(state)

        # Extract the number of remaining coins from the additional state information
        num_of_remaining_coins = add_state[0] if add_state is not None else 0

        # Choose between the distances
        # Take the distance to the nearest opponent if there is no remaining coin OR there is no crate OR the distance to the nearest opponent is smaller
        if num_of_remaining_coins == 0 or nearest_crate_distance is None or (nearest_opponent_distance is not None and nearest_opponent_distance < nearest_crate_distance):
            # Return the potential based on the distance to the nearest opponent
            # IMPORTANT: We add here + 1 to the negative of nearest_opponent_distance since without it, the
            #            the exponent can never get 0            
            return (1.2 ** (-nearest_opponent_distance + 1)) / 4 if nearest_opponent_distance is not None else 0.0
        else:
            # Return the potential based on the distance to the nearest crate
            # IMPORTANT: We add here + 1 to the negative of nearest_crate_distance since without it, the
            #            the exponent can never get 0            
            return (1.2 ** (-nearest_crate_distance + 1)) / 4 if nearest_crate_distance is not None else 0.0
    

def danger_potential_of_state(state: Union[dict, None]) -> float:
    """ Calculate the danger potential of the state.
        IMPORTANT: For the danger potential, we must use a discount factor of 1,
                   otherwise the agent gets a positive reward for entering and leaving
                   the blast coordinates.

    Args:
        state (Union[dict, None]): The state to calculate the danger potential of.

    Returns:
        float: The danger potential of the state.
    """

    # Check if the state is None
    if state is None:
        return 0.0

    # Get the position of the agent
    agent_position = state["self"][3]

    danger_penalty = 0
    for bomb in state["bombs"]:
        # Calculate the (Manhattan) distance to the bomb
        distance_to_bomb = manhattan_distance(agent_position, bomb[0])
        # Only look at bombs where the agent could be in the blast coordinates
        if distance_to_bomb <= s.BOMB_POWER:
            # Get the blast coordinates of the bomb
            blast_coords = get_bomb_blast_coords(bomb[0][0], bomb[0][1], state["field"], s.BOMB_POWER)       
            for coord in blast_coords:
                # Check if there the agent is in the blast coordinates
                if agent_position[0] == coord[0] and agent_position[1] == coord[1]:
                    # Compute the base penalty, which depends on the bomb timer
                    # NB: The bomb timer goes from 3 down to 0
                    base_penalty = 0.025 + 0.005 * (s.BOMB_TIMER - 1 - bomb[1])
                    # Increase the base penalty linearly if the agent is closer to the bomb
                    penalty = base_penalty * (distance_to_bomb - (s.BOMB_POWER + 1))
                    # Update the danger penalty
                    # NB: Also consider bombs at the agent's position (which can only happen if the agent dropped the bomb)
                    if penalty < danger_penalty:
                        danger_penalty = penalty

    return danger_penalty