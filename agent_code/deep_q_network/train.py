from collections import namedtuple, deque
import pickle
from typing import List
import torch.nn.functional as F
import ExperienceReplayBuffer from .model.experience_replay_buffer
import events as e
from typing import Union
from .callbacks import state_to_features

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Initialize the experience replay memory
    self.buffer = ExperienceReplayBuffer(buffer_capacity=self.CONFIG["BUFFER_CAPACITY"], device=self.device)

    # Initalize the optimizer
    self.optimizer = torch.optim.Adam(self.local_q_network.parameters(), lr=self.CONFIG["LEARNING_RATE"])

    # Initialize the target Q-network
    self.target_q_network = Network(state_size=self.CONFIG["STATE_SIZE"], action_size=self.CONFIG["ACTION_SIZE"]).to(self.device)

    # Load the weights of the local Q-network to the target Q-network
    self.target_q_network.load_state_dict(self.local_q_network.state_dict())

    # Initialize the exploration rate
    self.exploration_rate = self.CONFIG["EXPLORATION_RATE_START"]

    # Initialize the total number of steps
    self.total_steps = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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

    # Handle the step
    self.handle_step(state=state_to_features(old_game_state), action=self_action, reward=reward_from_events(events), next_state=state_to_features(new_game_state))

    self.logger.debug(f"Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}")

    # TODO: Add your own events to hand out rewards
    if False:
        events.append(PLACEHOLDER_EVENT)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    # Handle the step
    self.handle_step(state=state_to_features(old_game_state), action=self_action, reward=reward_from_events(events), next_state=None)

    self.logger.debug(f"Encountered event(s) {", ".join(map(repr, events))} in final step")

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


def handle_step(self, state: np.ndarray, action: str, reward: float, next_state: Union[np.ndarray, None]) -> None:
    """ Handle a step in the environment.

    Args:
        state (np.ndarray): The current state of the environment.
        action (str): The action taken in the state.
        reward (float): The reward received after taking the action.
        next_state (Union[np.ndarray, None]): The next state of the environment. None if the game has ended.
    """

    # Store the experience in the buffer
    self.buffer.push(state=state, action=action, reward=reward, next_state=next_state)

    # Update the exploration rate
    self.exploration_rate = update_exploration_rate(self.exploration_rate)

    # Increment the total number of steps
    self.total_steps += 1

    # Check if the agent should be trained and if the buffer has enough experiences to sample a batch
    if self.total_steps % self.CONFIG["TRAINING_FREQUENCY"] and len(self.buffer) > self.CONFIG["BATCH_SIZE"]:
        # Train the model
        self.train()

    #  Check if the target Q-network should be updated
    if self.total_steps % self.CONFIG["TARGET_UPDATE_FREQUENCY"] == 0:
        self.update_target_network()


def train(self) -> None:
    """ Train the local Q-network. """

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
    loss = F.mse_loss(q_expected, target_q_values)
    # Reset the gradients
    optimizer.zero_grad()
    # Peform the backpropagation
    loss.backward()
    # Update the weights
    optimizer.step()


def update_target_network(self) -> None:
    """ Update the target Q-network. """

    self.logger.debug(f"Updating target Q-network in round {last_game_state["round"]}")
    # TODO: Try soft update instead of hard update
    self.target_q_network.load_state_dict(self.local_q_network.state_dict()) 


def update_exploration_rate(self, exploration_rate: float) -> float:
    """ Update the exploration rate.

    Args:
        exploration_rate (float): The current exploration rate.

    Returns:
        float: The updated exploration rate.
    """
    
    # Decay the exploration rate if it is above the minimum exploration rate
    return max(self.CONFIG["EXPLORATION_RATE_MIN"], exploration_rate * self.CONFIG["EXPLORATION_DECAY"])

    
