import os
import torch
import wandb
import numpy as np
from collections import namedtuple, deque
from typing import Union, List
import events as e
import settings as s
from .model.experience_replay_buffer import ExperienceReplayBuffer
from .model.network import Network
from .callbacks import state_to_features
from .utils import round_ended_but_not_dead, set_seed, unset_seed, save_data, load_training_data, potential_of_state


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
        hidden_layer_size=self.CONFIG["HIDDEN_LAYER_SIZE"]
        ).to(self.device)
    # Load the weights of the local Q-network to the target Q-network
    self.target_q_network.load_state_dict(self.local_q_network.state_dict())
    # Initialize the experience replay memory
    self.buffer = ExperienceReplayBuffer(buffer_capacity=self.CONFIG["BUFFER_CAPACITY"], device=self.device)
    # Initalize the optimizer
    if self.CONFIG["OPTIMIZER"] == "Adam":
        self.optimizer = torch.optim.Adam(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    elif self.CONFIG["OPTIMIZER"] == "SGD":
        self.optimizer = torch.optim.SGD(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    elif self.CONFIG["OPTIMIZER"] == "RMSprop":
        self.optimizer = torch.optim.RMSprop(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    else:
        raise ValueError(f"Optimizer {self.CONFIG["OPTIMIZER"]} not supported")

    # Initialize the loss function
    self.loss_function = torch.nn.MSELoss()
    
    # Check if we can start from a saved state
    if "START_FROM" in self.CONFIG and self.CONFIG["START_FROM"] in ["best", "last"] and os.path.exists(f"{self.CONFIG["PATH"]}/{self.CONFIG["START_FROM"]}/"):
        print(f"Loading {self.CONFIG["START_FROM"]} training data and buffer from saved state.")
        self.exploration_rate, self.test_best_avg_score, self.training_steps, self.training_rounds = load_training_data(
            optimizer=self.optimizer, 
            buffer=self.buffer, 
            path=f"{self.CONFIG["PATH"]}/{self.CONFIG["START_FROM"]}/", 
            device=self.device)
    # Otherwise, set up the model from scratch
    else:
        print("Setting up training data and buffer from scratch.")
        # Initialize the exploration rate
        self.exploration_rate = self.CONFIG["EXPLORATION_RATE_START"]
        # Initialize the number of steps performed in training (exluding testing)
        self.training_steps = 0
        # Initialize the number of rounds performed in training (exluding testing)
        self.training_rounds = 0
        # Initialize the best average score achieved in the testing phase
        self.test_best_avg_score = 0

    # Initialize the reward per round/game during training
    self.training_reward_of_round = 0
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
    # Initialize the total steps in the testing phase
    self.test_total_steps = 0
    # Initialize the testing seed (so that each testing phase is is run on the same set of rounds/games)
    self.test_seed = self.CONFIG["TEST_SEED"]

    # Tell wandb to watch the model
    wandb.watch(self.local_q_network, self.loss_function, log="all", log_freq=10)

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

    # TODO: Add your own events to hand out rewards
    if False:
        events.append(PLACEHOLDER_EVENT)

    #self.logger.debug(f"Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}")

    # Handle the step
    # NB: Check if the the game/round has ended but the agent is not dead.
    #     In this case, the last step is already handled in end_of_round and thus should be ignored in game_events_occurred (this is a mistake in the environment)
    #     Note that the downside of this fix is that the event SURVIVED_ROUND is not always handled and should therefore not be used
    if not round_ended_but_not_dead(self, game_state=new_game_state): 
        handle_step(self, 
            state=old_game_state,
            action=self_action, 
            reward=get_reward(self, state=old_game_state, action=self_action, next_state=new_game_state, events=events),
            next_state=new_game_state,
            score=new_game_state["self"][1],
            change_world_seed=new_game_state["change_world_seed"],
            is_end_of_round=False
    )


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
    reward=get_reward(self, state=last_game_state, action=last_action, next_state=None, events=events),
    next_state=None,
    score=last_game_state["self"][1],
    change_world_seed=last_game_state["change_world_seed"],
    is_end_of_round=True)


def get_reward(self, state: dict, action: str, next_state: Union[dict, None], events: List[str]) -> int:
    """ Compute the reward based on the changed states and the events that occurred.

    Args:
        state (dict): The state before the action was taken.
        action (str): The action taken in the state.
        next_state (Union[dict, None]): The state after the action was taken. None if the round/game has ended by death of the agent.
        events (List[str]): The events that occurred when going from the state to the next state.

    Returns:
        int: The reward for the action taken in the state.
    """

    # Define the rewards for the events
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -0.1,
        e.WAITED: -0.1,
        e.BOMB_DROPPED: -0.5,
        # Penalize the agent for dying by the number of coins left (normalized)
        e.GOT_KILLED: -len(state["coins"])/s.SCENARIOS["coin-heaven"]["COIN_COUNT"] 
    }

    # Compute the reward based on the events
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # Add reward shaping based on the potential of the state and the next state
    reward_sum += self.CONFIG["DISCOUNT_RATE"] * potential_of_state(next_state) - potential_of_state(state)    

    # Log the reward and the events
    #self.logger.info(f"Awarded {reward_sum} for potential difference and for events {", ".join(events)}")
    
    return reward_sum


def handle_step(self, state: dict, action: str, reward: float, next_state: Union[dict, None], score: int, change_world_seed, is_end_of_round: bool = False) -> None:
    """ Handle a training step in the environment.

    Args:
        state (dict): The current state of the environment.
        action (str): The action taken in the state.
        reward (float): The reward received after taking the action.
        next_state (Union[dict, None]): The next state of the environment. None if the round/game has ended by death of the agent.
        score (int): The score of the agent in the next state.
        change_world_seed (function): Function to change the seed of the world.
        is_end_of_round (bool, optional): Whether the round/game has ended. Defaults to False.
    """

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
            self.test_total_steps += state["step"]
            # Check if the testing phase is over
            if self.test_rounds == self.CONFIG["ROUNDS_PER_TEST"]:
                self.test_training = False
                # Unset the seed for the testing phase
                unset_seed(change_world_seed, self.use_cuda)
                # Compute the average reward, score, and steps in the testing phase
                test_avg_reward = self.test_total_reward / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_score = self.test_total_score / self.CONFIG["ROUNDS_PER_TEST"]
                test_avg_steps = self.test_total_steps / self.CONFIG["ROUNDS_PER_TEST"]
                # Log the results of the testing phase
                wandb.log({ "test_avg_reward": test_avg_reward, "test_avg_score": test_avg_score, "test_avg_steps": test_avg_steps }, step=self.training_steps)
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
                    path=self.CONFIG["PATH"])
    # Otherwise the agent is in training mode
    else:
        # Store the experience in the buffer
        self.buffer.push(state=state_to_features(state), action=action, reward=reward, next_state=state_to_features(next_state))
        # Update the exploration rate
        self.exploration_rate = update_exploration_rate(self, self.exploration_rate)
        # Increment the total number of steps performed in training
        self.training_steps += 1
        # Increment the reward of the round/game during training
        self.training_reward_of_round += reward
        # Check if the agent should be trained and if the buffer has enough experiences to sample a batch
        if self.training_steps % self.CONFIG["TRAINING_FREQUENCY"] == 0 and len(self.buffer) > self.CONFIG["BUFFER_MIN_SIZE"]:
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
            # Log the stats of the round/game during training
            wandb.log({ 
                "training_reward_of_round": self.training_reward_of_round,
                "training_score_of_round": score, 
                "training_steps_of_round": state["step"],
                "training_round": self.training_rounds,
                "exploration_rate": self.exploration_rate,
                }, step=self.training_steps)
            # Reset the reward of the round/game during training
            self.training_reward_of_round = 0
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
                # Reset the total steps in the testing phase
                self.test_total_steps = 0
                # Set the seed for the testing phase
                set_seed(self.test_seed, change_world_seed, self.use_cuda)


def update_exploration_rate(self, exploration_rate: float) -> float:
    """ Update the exploration rate.

    Args:
        exploration_rate (float): The current exploration rate.

    Returns:
        float: The updated exploration rate.
    """
    
    # Decay the exploration rate if it is above the minimum exploration rate
    return max(self.CONFIG["EXPLORATION_RATE_MIN"], exploration_rate * self.CONFIG["EXPLORATION_RATE_DECAY"])


def train_network(self) -> None:
    """ Train the local Q-network. """

    # Set the local network to training mode
    self.local_q_network.train()
    # Sample a batch of experiences from the buffer
    states, actions, rewards, next_states, dones = self.buffer.sample(batch_size=self.CONFIG["BATCH_SIZE"])
    # Get the Q-values of the taken actions for the states from the local Q-network
    # NB: Add a dummy dimension with unsqueeze(1) for the actions to obtain the shape (batch_size, 1)
    #     This is necessary because .gather requires the shape of the actions to match the shape of the output of the local Q-network
    # NB: Remove the dummy dimension with squeeze(1) to obtain the shape (batch_size, ) for q_values  
    q_values = self.local_q_network(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
    # Get the maximum of the Q-values of the actions for the next states from the target Q-network
    # NB: Detach the tensor to prevent backpropagation through the target Q-network
    # The shape of next_q_values_max is (batch_size, )
    next_q_values_max = self.target_q_network(next_states).detach().max(dim=1)[0]        
    # Calculate the target Q-values
    target_q_values = rewards + self.CONFIG["DISCOUNT_RATE"] * next_q_values_max * (1 - dones)
    # Compute the MSE loss
    loss = self.loss_function(q_values, target_q_values)
    # Reset the gradients
    self.optimizer.zero_grad()
    # Peform the backpropagation
    loss.backward()
    # Log the loss
    wandb.log({"loss": loss.item()}, step=self.training_steps)
    # Update the weights
    self.optimizer.step()
    # Set the local network back to evaluation mode
    self.local_q_network.eval()


def update_target_network(self) -> None:
    """ Update the target Q-network. """

    #self.logger.debug(f"Updating target Q-network.")
    # TODO: Try soft update instead of hard update
    self.target_q_network.load_state_dict(self.local_q_network.state_dict()) 