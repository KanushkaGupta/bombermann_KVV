
import numpy as np
import os
import pickle
from .callbacks import state_to_features, update_q_table

def setup_training(self):
    """
    Initialize the training environment and log the setup.
    """
    self.logger.info("Training setup complete.")

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Handle game events that occurred after an action was taken.
    Assign rewards based on events and update the Q-Table accordingly.
    """
    reward = 0
    # Define rewards based on events
    for event in events:
        if event == 'COIN_COLLECTED':
            reward += 10
        if event == 'KILLED_OPPONENT':
            reward += 50
        if event == 'KILLED_SELF' or event == 'GOT_KILLED':
            reward -= 100  # Increased penalty
        if event == 'BOMB_DROPPED':
            reward += -1  # Small penalty for using bomb
        if event == 'WAITED':
            reward += -0.1  # Small penalty for waiting
        if event == 'CRATE_DESTROYED':
            reward += 2  # Small reward for destroying a crate
        if event == 'SAFE_MOVE':
            reward += 1  # Reward for making a safe move
        if event == 'DANGER_NEARBY':
            reward -= 5  # Penalty for moving into danger

    # Update Q-Table with the reward
    if self.last_state is not None and self.last_action is not None:
        next_state_features = state_to_features(new_game_state)
        if next_state_features is not None:
            next_state = tuple(next_state_features)
            update_q_table(self, self.last_state, self.last_action, reward, next_state)

def end_of_round(self, last_game_state, last_action, events):
    """
    Handle the end of a game round. Assign terminal rewards and save the Q-Table.
    """
    # Assign a terminal reward
    reward = 0
    if 'SURVIVED_ROUND' in events:
        reward += 20
    if 'OPPONENT_ELIMINATED' in events:
        reward += 30
    if 'GOT_KILLED' in events:
        reward -= 100  # Increased penalty

    if self.last_state is not None and self.last_action is not None:
        terminal_state_features = state_to_features(last_game_state)
        if terminal_state_features is not None:
            terminal_state = tuple(terminal_state_features)
            update_q_table(self, self.last_state, self.last_action, reward, terminal_state)

    # Save Q-Table
    with open(self.q_table_path, 'wb') as f:
        pickle.dump(dict(self.q_table), f)
    self.logger.info("Q-Table saved.")

# Register the functions to be accessible by the framework
train_callbacks = {
    "setup_training": setup_training,
    "game_events_occurred": game_events_occurred,
    "end_of_round": end_of_round
}

def train_setup_training(*args, **kwargs):
    train_callbacks["setup_training"](*args, **kwargs)

def train_game_events_occurred(*args, **kwargs):
    train_callbacks["game_events_occurred"](*args, **kwargs)

def train_end_of_round(*args, **kwargs):
    train_callbacks["end_of_round"](*args, **kwargs)
