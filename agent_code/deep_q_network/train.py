import os
import torch
import wandb
import time
import sys
import math
import numpy as np
from collections import namedtuple, deque
from typing import Union, List
import events as e
import settings as s
from .model.experience_replay_buffer import ExperienceReplayBuffer
from .model.network import Network
from .callbacks import state_to_features
from .utils import (
    round_ended_but_not_dead, 
    set_seed, 
    unset_seed, 
    save_data, 
    load_training_data, 
    potential_of_state,
    danger_potential_of_state, 
    num_crates_and_opponents_in_blast_coords,
    agent_has_trapped_itself,
    action_index_to_str,
    agents_waits_uselessly,
    end_reason_str_to_index
)

# Custom events
USELESS_BOMB = "USELESS_BOMB"
USEFUL_BOMB = "USEFUL_BOMB"
TRAPPED_ITSELF = "TRAPPED_ITSELF"
USELESS_WAIT = "USELESS_WAIT"


def setup_training(self) -> None:
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Initialize the target Q-network
    self.target_q_network = Network(
        channel_size=self.CONFIG["CHANNEL_SIZE"],
        column_size=(s.COLS - 2), 
        row_size=(s.ROWS - 2), 
        action_size=self.CONFIG["ACTION_SIZE"],
        hidden_layer_size=self.CONFIG["HIDDEN_LAYER_SIZE"],
        add_state_size=self.CONFIG["ADD_STATE_SIZE"],
        use_dueling_dqn=self.CONFIG["USE_DUELING_DQN"]
        ).to(self.device)
    # Load the weights of the local Q-network to the target Q-network
    self.target_q_network.load_state_dict(self.local_q_network.state_dict())
    # Initialize the experience replay memory
    self.buffer = ExperienceReplayBuffer(
        capacity=self.CONFIG["BUFFER_CAPACITY"], 
        device=self.device,
        discount_rate=self.CONFIG["DISCOUNT_RATE"],
        use_per=self.CONFIG["USE_PER"],
        transform_batch_randomly=self.CONFIG["TRANSFORM_BATCH_RANDOMLY"],
        n_steps=self.CONFIG["TD_STEPS"],
        priority_importance=self.CONFIG["PER_PRIORITY_IMPORTANCE"])
    # Initalize the optimizer
    if self.CONFIG["OPTIMIZER"] == "Adam":
        self.optimizer = torch.optim.Adam(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    elif self.CONFIG["OPTIMIZER"] == "SGD":
        self.optimizer = torch.optim.SGD(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    elif self.CONFIG["OPTIMIZER"] == "RMSprop":
        self.optimizer = torch.optim.RMSprop(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    else:
        raise ValueError(f"Optimizer {self.CONFIG["OPTIMIZER"]} not supported")
    
    # Check if we can start from a saved state
    if "START_FROM" in self.CONFIG and self.CONFIG["START_FROM"] in ["best", "last"] and os.path.exists(f"{self.CONFIG["PATH"]}/{self.CONFIG["START_FROM"]}/"):
        print(f"Loading {self.CONFIG["START_FROM"]} training data and buffer from saved state.")
        self.exploration_rate, self.weight_importance, self.test_best_avg_score, self.training_steps, self.training_rounds = load_training_data(
            optimizer=self.optimizer, 
            buffer=self.buffer, 
            path=f"{self.CONFIG["PATH"]}/{self.CONFIG["START_FROM"]}/", 
            device=self.device,
            use_per=self.CONFIG["USE_PER"])
    # Otherwise, set up the model from scratch
    else:
        print("Setting up training data and buffer from scratch.")
        # Initialize the exploration rate
        self.exploration_rate = self.CONFIG["EXPLORATION_RATE_START"]
        # Initialize the PER weight importance (ONLY USED FOR PRIORITIZED EXPERIENCE REPLAY)
        self.weight_importance = self.CONFIG["PER_WEIGHT_IMPORTANCE_START"] if self.CONFIG["USE_PER"] else None
        # Initialize the number of steps performed in training (exluding testing)
        self.training_steps = 0
        # Initialize the number of rounds performed in training (exluding testing)
        self.training_rounds = 0
        # Initialize the best average score achieved in the testing phase
        self.test_best_avg_score = 0

    # Initialize the reward per round/game during training
    self.training_reward_of_round = 0
    # Initialize the numbers of crates destroyed per round/game during training
    self.training_destroyed_crates_of_round = 0
    # Initialize the numbers of kills per round/game during training
    self.training_kills_of_round = 0
    # Initialize the number of bombs dropped per round/game during training
    self.training_dropped_bombs_of_round = 0
    # Initialize the number of loops broken per round/game during training
    if self.CONFIG["BREAK_LOOPS"]:
        self.training_broken_loops_of_round = 0
    # Initialize the start time of the training
    self.training_start_time = time.time()
    # Initialize the list of opponents encountered per round/game (used for computing the potential best other score)
    self.all_opponents_of_round = []
    # Initialize whether the agent is tested during training on a set of rounds/games with a fixed seed 
    self.test_training = False
    # Initialize whether the testing during training should be started in the next game/round
    self.start_test_training_next_round = False
    # Initialize the number of rounds/games performed in a testing phase during training (reset to 0 after each testing phase)
    self.test_rounds = 0
    # Initialize the total reward in the testing phase
    self.test_total_reward = 0
    # Initialize the total score in the testing phase
    self.test_total_score = 0
    # Initialize the total kills in the testing phase
    self.test_total_kills = 0
    # Initialize the total steps in the testing phase
    self.test_total_steps = 0
    # Initialize the total so far best other score in the testing phase
    self.test_total_so_far_best_other_score = 0
    # Initialize the total remaining score in the testing phase
    self.test_total_remaining_score_for_agent = 0
    # Initialize the testing seed (so that each testing phase is is run on the same set of rounds/games)
    self.test_seed = self.CONFIG["TEST_SEED"]

    # Tell wandb to watch the model
    wandb.watch(self.local_q_network, log="all", log_freq=10)

    # Store a reference to the W&B api
    self.wandbAPI = wandb.Api()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]) -> None:
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    #self.logger.debug(f"Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}")

    # Handle the step
    # NB: Check if the the game/round has ended but the agent is not dead.
    #     In this case, the last step is already handled in end_of_round and thus should be ignored in game_events_occurred (this is a mistake in the environment)
    #     Note that the downside of this fix is that the event SURVIVED_ROUND is not always handled and should therefore not be used
    if not round_ended_but_not_dead(self, game_state=new_game_state): 
        handle_step(self, 
            state=old_game_state,
            action=self_action, 
            next_state=new_game_state,
            events=events,
            score=old_game_state["self"][1],
            change_world_seed=new_game_state["change_world_seed"])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]) -> None:
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    #self.logger.debug(f"Encountered event(s) {", ".join(map(repr, events))} in final step")

    # Handle the step
    handle_step(self, 
    state=last_game_state,
    action=last_action,
    next_state=None,
    events=events,
    score=last_game_state["self"][1],
    change_world_seed=last_game_state["change_world_seed"])


def get_reward(self, state: dict, add_state: np.ndarray, action: str, next_state: Union[dict, None], add_next_state: Union[np.ndarray, None], events: List[str]) -> int:
    """ Compute the reward based on the changed states and the events that occurred.

    Args:
        state (dict): The state before the action was taken.
        add_state (np.ndarray): The additional state before the action was taken.
        action (str): The action taken in the state.
        next_state (Union[dict, None]): The state after the action was taken. None if the round/game has ended.
        add_next_state (Union[np.ndarray, None]): The additional next state after the action was taken. None if the round/game has ended.
        events (List[str]): The events that occurred when going from the state to the next state.

    Returns:
        int: The reward for the action taken in the state.
    """

    # Check if the agent performs a useless wait
    if agents_waits_uselessly(state, action, next_state, events):
        events.append(USELESS_WAIT)

    # Initialize the number of crates attacked by the dropped bomb
    # NB: This variable is only used when BOMB_DROPPED is in events
    num_crates_attacked = 0
    # Initialize the number of opponents attacked by the dropped bomb
    # NB: This variable is only used when BOMB_DROPPED is in events
    num_opponents_attacked = 0

    # Check if the agent has dropped a bomb
    if action == "BOMB" and e.BOMB_DROPPED in events:
        # Get the number of crates attacked by the dropped bomb
        num_crates_attacked, num_opponents_attacked = num_crates_and_opponents_in_blast_coords(np.array(state["self"][3]), state["field"], state["others"])
        # Check if the dropped bomb is useless
        if num_crates_attacked == 0 and num_opponents_attacked == 0:
            events.append(USELESS_BOMB)
        # Otherwise the dropped bomb is useful
        else:
            events.append(USEFUL_BOMB)

    # Check if the agent has just trapped itself by moving away from its own dropped bomb
    if agent_has_trapped_itself(state, action, next_state, events):
        events.append(TRAPPED_ITSELF)

     # Extract the number of remaining coins from the additional state information
    num_of_remaining_coins = add_state[0] if add_state is not None else 0

    # Define the rewards for the events
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -0.1,
        USELESS_WAIT: -0.1,
        # NB: ONLY IF USE_DANGER_POTENTIAL: The agent receives a penalty of -0.1 for placing a bomb due to the potential so we compensate this in USEFUL_BOMB
        # NB2: The agent receives a penalty of at most -0.25 for dropping down in the crate potential function since now the distance to 
        #      another crate is used, which is further away. Note that the agent drops at least -0.09 because the next crate that is 
        #      not in a blast coordinate and for which the last tile of the path is also not in a blast coordinate is at least 4 steps
        #      away (crate_pot(4) - crate_pot(1) = -0.09). Besides, in a 11 x 11 field, the max distance to a crate is 16 and crate_pot(16) = 0.195.
        #      Thus, the total possible range of USEFUL_BOMB is 0.115 [= 0.25 + 0.05 - 0.195 + 0.01 + 0] to 0.41 [= 0.25 + 0.05 - 0.09 + 0.1 + 0.1] depending on the number of crates and opponents attacked
        USEFUL_BOMB: 0.25 + 0.05 + 0.1 * (num_crates_attacked / 10) + 0.1 * (num_opponents_attacked / self.CONFIG["NUM_OPPONENTS"]) + (0.1 if self.CONFIG["USE_DANGER_POTENTIAL"] else 0.0),
        # NB: ONLY IF USE_DANGER_POTENTIAL: Similar to USEFUL_BOMB, the agent receives a penalty of -0.1 for placing a bomb due to the potential but we do NOT compensate this in USELESS_BOMB, since
        #     it should remain a penalty (negative)
        # NB2: Contrary to USEFUL_BOMB, the crate potential function does not drop down (since no crate attacked) and thus it does NOT have to be compensated.
        USELESS_BOMB: -0.1 if self.CONFIG["USE_DANGER_POTENTIAL"] else -0.15,
        TRAPPED_ITSELF: -0.5,
        # Penalize the agent for dying by the number of coins and opponents left (normalized)
        e.GOT_KILLED: -(s.REWARD_COIN * num_of_remaining_coins + s.REWARD_KILL * len(state["others"]))/(s.REWARD_COIN * s.SCENARIOS["classic"]["COIN_COUNT"] + s.REWARD_KILL * self.CONFIG["NUM_OPPONENTS"]),
    }

    # Compute the reward based on the events
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]


    # Add reward shaping based on the potential of the state and the next state
    reward_sum += self.CONFIG["DISCOUNT_RATE"] * potential_of_state(next_state, add_next_state) - potential_of_state(state, add_state)    

    # Add reward shaping based on the danger potential of the state and the next state
    # IMPORTANT: For the danger potential, we must use a discount factor of 1,
    #            otherwise the agent gets a positive reward for entering and leaving
    #            the blast coordinates.
    if self.CONFIG["USE_DANGER_POTENTIAL"]:
        reward_sum += danger_potential_of_state(next_state) - danger_potential_of_state(state)    

    # Log the reward and the events
    #self.logger.info(f"Awarded {reward_sum} for potential difference and for events {", ".join(events)}")
    
    return reward_sum


def handle_step(self, state: dict, action: str, next_state: Union[dict, None], events: List[str], score: int, change_world_seed) -> None:
    """ Handle a training step in the environment.

    Args:
        state (dict): The current state of the environment.
        action (str): The action taken in the state.
        next_state (Union[dict, None]): The next state of the environment. None if the round/game has ended.
        events (List[str]): The events that occurred when going from the state to the next state.
        score (int): The score of the agent in the state (before the action was taken).
        change_world_seed (function): Function to change the seed of the world.
    """

    # The round/game has ended if the next state is None
    is_end_of_round = (next_state == None)    

    # Get the score obtained by making the action in the state
    score_of_action = events.count(e.COIN_COLLECTED) * s.REWARD_COIN + events.count(e.KILLED_OPPONENT) * s.REWARD_KILL
    # Extend the score based on the score of the action
    score += score_of_action

    # Set the additional state (which is the number of remaining coins and the number of remaining opponents)
    add_state = np.array([self.num_of_remaining_coins / s.SCENARIOS["classic"]["COIN_COUNT"], len(state["others"]) / self.CONFIG["NUM_OPPONENTS"]])
    # Initialize the next number of remaining coins
    next_num_of_remaining_coins = self.num_of_remaining_coins if next_state is not None else 0
    # Initialize the number of collected coins
    collected_coins = 0
    # Loop over the coins of the current game state
    #assert self.prev_coins == state["coins"]
    if next_state is not None:
        for coin in state["coins"]:
            # Check if the coin is not anymore in the next game state
            if coin not in next_state["coins"]:
                collected_coins += 1
    # Update the number of remaining coins
    next_num_of_remaining_coins -= collected_coins
    # Set the additional next state (which is the next number of remaining coins and the next number of remaining opponents)
    add_next_state = np.array([next_num_of_remaining_coins / s.SCENARIOS["classic"]["COIN_COUNT"], len(next_state["others"]) / self.CONFIG["NUM_OPPONENTS"]]) if next_state is not None else None

    # Comput the reward for the action taken in the state
    reward = get_reward(self, state=state, add_state=add_state, action=action, next_state=next_state, add_next_state=add_next_state, events=events)

    # Update the list of opponents encountered in the round/game
    # NB: We use state instead of next_state, since if end_of_round is True, next_state is None and state is the last state of the round/game
    if len(self.all_opponents_of_round) == 0:
        self.all_opponents_of_round = state["others"]
    else:
        # Loop over the opponents in the current state
        for opponent in state["others"]:
            # Loop over the stored opponents in the round/game
            for index, stored_opponent in enumerate(self.all_opponents_of_round):
                # Find the opponent in the stored opponents
                if opponent[0] == stored_opponent[0]:
                    # Update the stored opponent
                    self.all_opponents_of_round[index] = opponent
                    # Break the inner loop
                    break

    # Check if the agent is tested during training
    if self.test_training:
        # Add the reward to the total reward
        self.test_total_reward += reward
        # Check if the round/game has ended
        if is_end_of_round:
            # Increment the number of testing rounds
            self.test_rounds += 1
            # Add the score to the total score
            self.test_total_score += score
            # Add the kills to the total kills
            self.test_total_kills += events.count(e.KILLED_OPPONENT)
            # Add the steps to the total steps
            self.test_total_steps += state["step"]
            # Add the so far best other score to the total potential best other score
            self.test_total_so_far_best_other_score += get_so_far_best_other_score(self, events)
            # Add the remaining score for the agent to the total remaining score
            self.test_total_remaining_score_for_agent += s.REWARD_COIN * self.num_of_remaining_coins + s.REWARD_KILL * len(state["others"]) - score_of_action
            # Check if the testing phase is over
            if self.test_rounds == self.CONFIG["ROUNDS_PER_TEST"]:
                self.test_training = False
                # Unset the seed for the testing phase
                unset_seed(change_world_seed, self.use_cuda)
                # Compute the average reward, score, and steps in the testing phase
                test_avg_reward = self.test_total_reward / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_score = self.test_total_score / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_kills = self.test_total_kills / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_steps = self.test_total_steps / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_so_far_best_other_score = self.test_total_so_far_best_other_score / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_remaining_score_for_agent = self.test_total_remaining_score_for_agent / self.CONFIG["ROUNDS_PER_TEST"]
                # Log the results of the testing phase
                wandb.log({ 
                    "test_avg_reward": test_avg_reward,
                    "test_avg_score": test_avg_score,
                    "test_avg_kills": test_avg_kills,
                    "test_avg_steps": test_avg_steps,
                    "test_avg_so_far_best_other_score": test_avg_so_far_best_other_score,
                    "test_avg_remaining_score_for_agent": test_avg_remaining_score_for_agent,
                }, step=self.training_steps)
                # Check if the average score is higher than the best score
                is_test_best_avg_score = False
                if test_avg_score > self.test_best_avg_score:
                    # Update the best score
                    self.test_best_avg_score = test_avg_score
                    is_test_best_avg_score = True
                # Create the metadata
                metadata = {
                    "test_avg_reward": test_avg_reward,
                    "test_avg_score": test_avg_score,
                    "is_test_best_avg_score": is_test_best_avg_score,
                    "exploration_rate": self.exploration_rate,
                    "weight_importance": self.weight_importance if self.CONFIG["USE_PER"] else None,
                    "training_steps": self.training_steps,
                    "training_rounds": self.training_rounds,
                    "config": self.CONFIG
                }
                # Save the model
                print("Save the model") 
                save_data(
                    project_name=self.CONFIG["PROJECT_NAME"],
                    run=self.run,
                    run_name=self.run_name,
                    wandbAPI=self.wandbAPI,
                    metadata=metadata,
                    network=self.local_q_network, 
                    optimizer=self.optimizer, 
                    buffer=self.buffer, 
                    test_best_avg_score=self.test_best_avg_score,
                    path=self.CONFIG["PATH"],
                    use_per=self.CONFIG["USE_PER"])
    # Otherwise the agent is in training mode
    else:
        # Store the experience in the buffer
        self.buffer.push(state=state_to_features(state), add_state=add_state, action=action, reward=reward, next_state=state_to_features(next_state), add_next_state=add_next_state)
        # Update the exploration rate
        self.exploration_rate = update_exploration_rate(self, self.exploration_rate)
        # Update the weights importance (ONLY USED FOR PRIORITIZED EXPERIENCE REPLAY)
        if self.CONFIG["USE_PER"]:
            self.weight_importance = update_weight_importance(self, self.weight_importance)
        # Increment the total number of steps performed in training
        self.training_steps += 1
        # Increment the reward of the round/game during training
        self.training_reward_of_round += reward
        # Increment the number of crates destroyed in the round/game during training
        self.training_destroyed_crates_of_round += events.count(e.CRATE_DESTROYED)
        # Increment the number of kills in the round/game during training
        self.training_kills_of_round += events.count(e.KILLED_OPPONENT)
        # Increment the number of dropped bombs in the round/game during training
        self.training_dropped_bombs_of_round += events.count(e.BOMB_DROPPED)
        # Check if the agent should be trained and if the buffer has enough experiences to sample a batch
        if self.training_steps % self.CONFIG["TRAINING_FREQUENCY"] == 0 and len(self.buffer) >= self.CONFIG["BUFFER_MIN_SIZE"]:
            # Train the model
            train_network(self)
        #  Check if the target Q-network should be updated
        if self.training_steps % self.CONFIG["TARGET_UPDATE_FREQUENCY"] == 0:
            update_target_network(self)
        # Check if the agent should be tested during training
        if self.training_steps % self.CONFIG["TESTING_FREQUENCY"] == 0:
            # Store that the testing phase is started in the next game/round
            self.start_test_training_next_round = True
        # Check if the round/game has ended
        if is_end_of_round:
            # Increment the number of rounds/games performed in training
            self.training_rounds += 1
            # Get the reason for the end of the round/game in training
            training_end_reason_of_round = e.SURVIVED_ROUND
            if e.KILLED_SELF in events:
                training_end_reason_of_round = e.KILLED_SELF
            elif e.GOT_KILLED in events:
                training_end_reason_of_round = e.GOT_KILLED
            # Get the so far best other score in the round/game during training
            training_so_far_best_other_score_of_round = get_so_far_best_other_score(self, events)
            # Get the remaining score for the agent in the round/game during training
            training_remaining_score_for_agent_of_round = s.REWARD_COIN * self.num_of_remaining_coins + s.REWARD_KILL * len(state["others"]) - score_of_action
            # Log the stats of the round/game during training
            wandb.log({ 
                "training_reward_of_round": self.training_reward_of_round,
                "training_destroyed_crates_of_round": self.training_destroyed_crates_of_round,
                "training_kills_of_round": self.training_kills_of_round,
                "training_end_reason_of_round": end_reason_str_to_index(training_end_reason_of_round),
                "training_so_far_best_other_score_of_round": training_so_far_best_other_score_of_round,
                "training_remaining_score_for_agent_of_round": training_remaining_score_for_agent_of_round,
                "training_dropped_bombs_of_round": self.training_dropped_bombs_of_round,
                "training_score_of_round": score, 
                "training_steps_of_round": state["step"],
                "training_round": self.training_rounds,
                "exploration_rate": self.exploration_rate,
                }, step=self.training_steps)

            # Log the weights importance (ONLY USED FOR PRIORITIZED EXPERIENCE REPLAY)
            if self.CONFIG["USE_PER"]:
                wandb.log({ "weight_importance": self.weight_importance }, step=self.training_steps)
            # Log the number of broken loops (if enabled)
            if self.CONFIG["BREAK_LOOPS"]:
                wandb.log({ "training_broken_loops_of_round": self.training_broken_loops_of_round }, step=self.training_steps)
            # Reset the reward of the round/game during training
            self.training_reward_of_round = 0
            # Reset the number of destroyed crates in the round/game during training
            self.training_destroyed_crates_of_round = 0
            # Reset the number of kills in the round/game during training
            self.training_kills_of_round = 0
            # Reset the number of dropped bombs in the round/game during training
            self.training_dropped_bombs_of_round = 0
            # Reset the list of opponents encountered in the round/game during training
            self.all_opponents_of_round = []
            # Clear the loop buffer and reset the number of broken loops in the round/game during training
            if self.CONFIG["BREAK_LOOPS"]:
                self.loop_buffer.clear()
                self.training_broken_loops_of_round = 0
            # Check if the maximum runtime is reached
            if "MAX_RUNTIME" in self.CONFIG and self.CONFIG["MAX_RUNTIME"] is not None and time.time() - self.training_start_time > self.CONFIG["MAX_RUNTIME"]:
                print("Maximum runtime reached.")
                # Stop the training
                sys.exit()
            # Check if the testing phase should be started in the next game/round
            if self.start_test_training_next_round:
                # Reset the flag
                self.start_test_training_next_round = False
                # Set the agent to testing mode
                self.test_training = True
                # Reset the number of testing rounds
                self.test_rounds = 0
                # Reset the total reward in the testing phase
                self.test_total_reward = 0
                # Reset the total score in the testing phase
                self.test_total_score = 0
                # Reset the total kills in the testing phase
                self.test_total_kills = 0
                # Reset the total steps in the testing phase
                self.test_total_steps = 0
                # Reset the total so far best other in the testing phase
                self.test_total_so_far_best_other_score = 0
                # Reset the total remaining score for the agent in the testing phase
                self.test_total_remaining_score_for_agent = 0
                # Set the seed for the testing phase
                set_seed(self.test_seed, change_world_seed, self.use_cuda)


def get_so_far_best_other_score(self, events: List[str]) -> int:
    """ Compute the so far best other score in the round/game.

    Args:
        events (List[str]): The events that occurred in the round/game.

    Returns:
        int: The so far best other score in the round/game.
    """
    # Determine the killer of the agent (if any)
    agent_killer = None
    if e.GOT_KILLED in events and not e.KILLED_SELF in events:
        agent_killer = [event for event in events if event.startswith("KILLED_BY")][0].split(":")[1]

    # Initialize the ccurrent best other score
    current_best_other_score = 0
    # Loop over the opponents in the round/game
    for opponent in self.all_opponents_of_round:
        # Check if the opponent has a higher score than the current best other score
        score = opponent[1] + (s.REWARD_KILL if agent_killer and opponent[0] == agent_killer else 0)
        if current_best_other_score < score:
            # Update the current best other score
            current_best_other_score = score

    return current_best_other_score


def update_exploration_rate(self, exploration_rate: float) -> float:
    """ Update the exploration rate.

    Args:
        exploration_rate (float): The current exploration rate.

    Returns:
        float: The updated exploration rate.
    """
    
    # Decay the exploration rate if it is above the minimum exploration rate
    return max(self.CONFIG["EXPLORATION_RATE_MIN"], exploration_rate * self.CONFIG["EXPLORATION_RATE_DECAY"])


def update_weight_importance(self, weight_importance: float) -> float:
    """ Update the weight importance.

    Args:
        weight_importance (float): The current weight importance.

    Returns:
        float: The updated weight imporrtance.
    """
    
    # Increment the weight importance if it is below the maximum weight importance
    return min(self.CONFIG["PER_WEIGHT_IMPORTANCE_MAX"], weight_importance + self.CONFIG["PER_WEIGHT_IMPORTANCE_INCREMENT"])


def train_network(self) -> None:
    """ Train the local Q-network. """

    # Set the local network to training mode
    self.local_q_network.train()
    for i in range(self.CONFIG["NUM_EPOCHS"]):
        # Sample a batch of experiences from the buffer
        states, add_states, actions, rewards, next_states, add_next_states, dones, indices, weights = self.buffer.sample(batch_size=self.CONFIG["BATCH_SIZE"], weight_importance=self.weight_importance)
        # Get the Q-values of the taken actions for the states from the local Q-network
        # NB: Add a dummy dimension with unsqueeze(1) for the actions to obtain the shape (batch_size, 1)
        #     This is necessary because .gather requires the shape of the actions to match the shape of the output of the local Q-network
        # NB: Remove the dummy dimension with squeeze(1) to obtain the shape (batch_size, ) for q_values  
        q_values = self.local_q_network(states, add_states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # Initialize the next Q-values
        next_q_values = None

        if self.CONFIG["USE_DOUBLE_DQN"]:
            # Use the local Q-network to select the best actions (with the highest Q-value) for the next states
            next_actions = self.local_q_network(next_states, add_next_states).argmax(dim=1)

            # Use the target Q-network to compute the Q-values of the next states for the actions chosen by the local Q-network
            # NB: Add a dummy dimension with unsqueeze(1) for the actions to obtain the shape (batch_size, 1)
            #     This is necessary because .gather requires the shape of the actions to match the shape of the output of the target Q-network
            # NB2: Detach the tensor to prevent backpropagation through the target Q-network
            # NB3: Remove the dummy dimension with squeeze(1) to obtain the shape (batch_size, ) for q_values  
            # The shape of next_q_values is (batch_size, )
            next_q_values = self.target_q_network(next_states, add_next_states).gather(dim=1, index=next_actions.unsqueeze(1)).detach().squeeze(1) 
        else: 
            # Get the maximum of the Q-values of the actions for the next states from the target Q-network
            # NB: Detach the tensor to prevent backpropagation through the target Q-network
            # The shape of next_q_values is (batch_size, )
            next_q_values = self.target_q_network(next_states, add_next_states).detach().max(dim=1)[0]        
        # Calculate the target Q-values
        target_q_values = rewards + self.CONFIG["DISCOUNT_RATE"] * next_q_values * (1 - dones)
        # Initialize the MSE loss
        loss = None
        # Initialize the updated priorities (ONLY USED FOR PRIORITIZED EXPERIENCE REPLAY)
        priorities = None
        # Check if the agent uses prioritized experience replay
        if self.CONFIG["USE_PER"]:
            # If so, multiply the loss by the weights before computing the mean
            loss = (q_values - target_q_values).pow(2) * weights
            # Compute the updated priorities (add a small value to the priorities to avoid zero priorities)
            priorities = loss + 1e-5
            # Compute the mean of the loss
            loss = loss.mean()
        else:
            # Otherwise, simply compute the MSE loss
            loss = (q_values - target_q_values).pow(2).mean()
        # Reset the gradients
        self.optimizer.zero_grad()
        # Peform the backpropagation
        loss.backward()
        # Log the loss
        wandb.log({"loss": loss.item()}, step=self.training_steps)
        # Update the priorities in the buffer (ONLY USED FOR PRIORITIZED EXPERIENCE REPLAY)
        if self.CONFIG["USE_PER"]:
            self.buffer.update_priorities(indices=indices, priorities=priorities.detach().cpu().numpy())
        # Update the weights
        self.optimizer.step()
    # Set the local network back to evaluation mode
    self.local_q_network.eval()


def update_target_network(self) -> None:
    """ Update the target Q-network. """

    #self.logger.debug(f"Updating target Q-network.")
    # TODO: Try soft update instead of hard update
    self.target_q_network.load_state_dict(self.local_q_network.state_dict()) 