from collections import namedtuple, deque
import pickle
from typing import Union, List
import torch
import torch.nn.functional as F
import numpy as np
import events as e
import settings as s
from .model.experience_replay_buffer import ExperienceReplayBuffer
from .model.network import Network
from .callbacks import state_to_features
from .utils import round_ended_but_not_dead, set_seed, unset_seed

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self) -> None:
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Initialize the experience replay memory
    self.buffer = ExperienceReplayBuffer(buffer_capacity=self.CONFIG["BUFFER_CAPACITY"], device=self.device)
    # Initialize the target Q-network
    self.target_q_network = Network(channel_size=self.CONFIG["CHANNEL_SIZE"], column_size=(s.COLS - 2), row_size=(s.ROWS - 2), action_size=self.CONFIG["ACTION_SIZE"]).to(self.device)
    # Load the weights of the local Q-network to the target Q-network
    self.target_q_network.load_state_dict(self.local_q_network.state_dict())
    # Initalize the optimizer
    self.optimizer = torch.optim.Adam(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])
    # Initialize the exploration rate
    self.exploration_rate = self.CONFIG["EXPLORATION_RATE_START"]
    # Initialize whether the agent is tested during training on a set of rounds/games with a fixed seed 
    self.test_training = False
    # Initialize whether the testing during training should be started in the next game/round
    self.start_test_training_next_round = False
    # Initialize the number of steps performed in training (exluding testing)
    self.training_steps = 0
    # Initialize the number of rounds/games performed in a testing phase during training (reset to 0 after each testing phase)
    self.testing_rounds = 0
    # Initialize the total reward in the testing phase
    self.testing_total_reward = 0
    # Initialize the total score in the testing phase
    self.testing_total_score = 0
    # Initialize the testing seed (so that each testing phase is is run on the same set of rounds/games)
    self.testing_seed = 1234


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

    self.logger.debug(f"Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}")

    # Handle the step
    handle_step(self, 
        state=state_to_features(old_game_state),
        action=self_action, 
        reward=reward_from_events(self, events=events),
        next_state=state_to_features(new_game_state),
        score=new_game_state["self"][1],
        change_world_seed=new_game_state["change_world_seed"],
        is_end_of_round=round_ended_but_not_dead(self, game_state=new_game_state)
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
    
    self.logger.debug(f"Encountered event(s) {", ".join(map(repr, events))} in final step")
    # Handle the step
    # NB: Check if the the game/round has ended but the agent is not dead.
    #     In this case, the last step is already handled in game_events_occurred and thus should be ignored in end_of_round (this is a mistake in the environment)
    #     Note that the downside of this fix is that the event SURVIVED_ROUND is not always handled and should therefore not be used
    if not round_ended_but_not_dead(self, game_state=last_game_state): 
        handle_step(self, 
        state=state_to_features(last_game_state),
        action=last_action,
        reward=reward_from_events(self, events=events),
        next_state=None,
        score=last_game_state["self"][1],
        change_world_seed=last_game_state["change_world_seed"],
        is_end_of_round=True)

    # TODO: Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {", ".join(events)}")
    return reward_sum


def handle_step(self, state: np.ndarray, action: str, reward: float, next_state: Union[np.ndarray, None], score: int, change_world_seed, is_end_of_round: bool = False) -> None:
    """ Handle a training step in the environment.

    Args:
        state (np.ndarray): The current state of the environment.
        action (str): The action taken in the state.
        reward (float): The reward received after taking the action.
        next_state (Union[np.ndarray, None]): The next state of the environment. None if the round/game has ended by death of the agent.
        score (int): The score of the agent in the next state.
        change_world_seed (function): Function to change the seed of the world.
        is_end_of_round (bool, optional): Whether the round/game has ended. Defaults to False.
    """

    # Check if the agent is tested during training
    if self.test_training:
        # Add the reward to the total reward
        self.testing_total_reward += reward
        # Check if the round/game has ended
        if is_end_of_round:
            # Increment the number of testing rounds
            self.testing_rounds += 1
            # Add the score to the total score
            self.testing_total_score += score
            # Check if the testing phase is over
            if self.testing_rounds == self.CONFIG["ROUNDS_PER_TEST"]:
                self.test_training = False
                # Unset the seed for the testing phase
                unset_seed(change_world_seed, self.use_cuda)
                # Print the results of the testing phase
                print("Average reward:", self.testing_total_reward / self.CONFIG["ROUNDS_PER_TEST"])
                print("Average score:", self.testing_total_score / self.CONFIG["ROUNDS_PER_TEST"])
    # Otherwise the agent is in training mode
    else:
        # Store the experience in the buffer
        self.buffer.push(state=state, action=action, reward=reward, next_state=next_state)
        # Update the exploration rate
        self.exploration_rate = update_exploration_rate(self, self.exploration_rate)
        # Increment the total number of steps
        self.training_steps += 1
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
        # Check if the testing phase should be started in the next game/round
        if self.start_test_training_next_round and is_end_of_round:
            # Reset the flag
            self.start_test_training_next_round = False
            # Set the agent to testing mode
            self.test_training = True
            # Reset the number of testing rounds
            self.testing_rounds = 0
            # Reset the total reward in the testing phase
            self.testing_total_reward = 0
            # Reset the total score in the testing phase
            self.testing_total_score = 0
            # Set the seed for the testing phase
            set_seed(self.testing_seed, change_world_seed, self.use_cuda)


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
    loss = F.mse_loss(q_values, target_q_values)
    # Reset the gradients
    self.optimizer.zero_grad()
    # Peform the backpropagation
    loss.backward()
    print(loss.item())
    # Update the weights
    self.optimizer.step()
    # Set the local network back to evaluation mode
    self.local_q_network.eval()


def update_target_network(self) -> None:
    """ Update the target Q-network. """

    self.logger.debug(f"Updating target Q-network.")
    # TODO: Try soft update instead of hard update
    self.target_q_network.load_state_dict(self.local_q_network.state_dict()) 