

import numpy as np
import random
from collections import defaultdict
import pickle
import os
from .utils import get_new_position, will_be_in_blast

def state_to_features(game_state: dict) -> np.ndarray:
    """
    Convert the game state to a discretized feature vector.
    """
    if game_state is None:
        return None

    # Initialize feature vector
    features = []

    # Extract your agent's information
    _, score, bomb_available, (x, y) = game_state['self']

    # 1. Surrounding tiles (up, right, down, left)
    field = game_state['field']
    surroundings = [
        field[x, y-1] if y > 0 else -1,      # Up
        field[x+1, y] if x < field.shape[0]-1 else -1,  # Right
        field[x, y+1] if y < field.shape[1]-1 else -1,  # Down
        field[x-1, y] if x > 0 else -1       # Left
    ]
    features.extend(surroundings)

    # 2. Bomb availability
    features.append(int(bomb_available))

    # 3. Distance to nearest coin (discretized)
    coins = game_state['coins']
    if coins:
        distances = [np.linalg.norm(np.array([x, y]) - np.array(coin)) for coin in coins]
        min_coin_distance = min(distances)
    else:
        min_coin_distance = -1  # No coins left

    # Discretize the distance to the nearest coin
    if min_coin_distance == -1:
        coin_distance_discrete = -1  # No coins
    elif min_coin_distance <= 2:
        coin_distance_discrete = 0  # Close
    elif min_coin_distance <= 5:
        coin_distance_discrete = 1  # Medium
    else:
        coin_distance_discrete = 2  # Far
    features.append(coin_distance_discrete)

    # 4. Danger level (e.g., bombs in the vicinity)
    explosion_map = game_state['explosion_map']
    danger_level = explosion_map[x, y]
    features.append(int(danger_level > 0))

    # 5. Safe directions (1 if safe, 0 otherwise)
    safe_directions = []
    for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        nx, ny = get_new_position((x, y), action)
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if explosion_map[nx, ny] == 0 and field[nx, ny] == 0:
                safe_directions.append(1)
            else:
                safe_directions.append(0)
        else:
            safe_directions.append(0)
    features.extend(safe_directions)  # +4 features

    # 6. Distance to nearest opponent (discretized)
    others = game_state['others']
    if others:
        opponent_positions = [opponent[-1] for opponent in others]
        opponent_distances = [np.linalg.norm(np.array([x, y]) - np.array(pos)) for pos in opponent_positions]
        min_opponent_distance = min(opponent_distances)
    else:
        min_opponent_distance = -1  # No opponents left

    # Discretize the distance to the nearest opponent
    if min_opponent_distance == -1:
        opponent_distance_discrete = -1  # No opponents
    elif min_opponent_distance <= 2:
        opponent_distance_discrete = 0  # Close
    elif min_opponent_distance <= 5:
        opponent_distance_discrete = 1  # Medium
    else:
        opponent_distance_discrete = 2  # Far
    features.append(opponent_distance_discrete)

    # 7. Number of active bombs (capped)
    bombs = game_state['bombs']
    num_active_bombs = min(len(bombs), 2)  # Cap at 2
    features.append(num_active_bombs)

    # 8. Number of opponents (capped)
    num_opponents = min(len(others), 3)  # Cap at 3
    features.append(num_opponents)

    # 9. Is the agent currently in danger?
    in_danger = int(will_be_in_blast((x, y), game_state, time_step=0))
    features.append(in_danger)

    # 10. Time until the nearest bomb explodes (discretized)
    if bombs:
        min_bomb_countdown = min([bomb[1] for bomb in bombs])
        if min_bomb_countdown <= 2:
            bomb_timer = 0  # Imminent
        else:
            bomb_timer = 1  # Later
    else:
        bomb_timer = -1  # No bombs
    features.append(bomb_timer)

    # 11. Number of escape routes (discretized)
    num_safe_directions = sum(safe_directions)
    features.append(num_safe_directions)

    # 12. Presence of own bomb in the vicinity (1 if yes, 0 otherwise)
    own_bomb_near = 0
    for bomb in bombs:
        if bomb[0] == (x, y):
            own_bomb_near = 1
            break
    features.append(own_bomb_near)

    # Ensure feature vector has consistent length
    while len(features) < 20:
        features.append(0)

    features_array = np.array(features, dtype=int)
    return features_array

def setup(self):
    """
    Initialize the Q-Learning agent's parameters and Q-Table.
    """
    # Initialize Q-Table as a defaultdict with zero arrays for each state
    self.q_table = defaultdict(lambda: np.zeros(6))  # 6 possible actions

    # Q-Learning parameters
    self.alpha = 0.1       # Learning rate
    self.gamma = 0.9       # Discount factor
    self.epsilon = 1.0     # Exploration rate
    self.epsilon_min = 0.1
    self.epsilon_decay = 0.995

    # Initialize last state and action
    self.last_state = None
    self.last_action = None

    # Path to save/load Q-Table based on agent's directory
    current_dir = os.path.dirname(__file__)
    self.q_table_path = os.path.join(current_dir, 'q_table.pkl')

    # Ensure the directory exists
    os.makedirs(current_dir, exist_ok=True)

    # Load existing Q-Table if available
    if os.path.exists(self.q_table_path):
        try:
            with open(self.q_table_path, 'rb') as f:
                loaded_q_table = pickle.load(f)
            # Convert loaded Q-Table to defaultdict
            self.q_table = defaultdict(lambda: np.zeros(6), loaded_q_table)
            self.logger.info("Loaded existing Q-Table.")
        except Exception as e:
            self.logger.error(f"Failed to load Q-Table: {e}")
            self.q_table = defaultdict(lambda: np.zeros(6))
    else:
        self.logger.info("Initialized new Q-Table.")

    self.logger.info("Q-Learning Agent setup complete.")

def update_q_table(self, state, action, reward, next_state):
    """
    Update the Q-Table using the Q-Learning update rule.
    """
    old_value = self.q_table[state][action]
    next_max = np.max(self.q_table[next_state])
    # Q-Learning update rule
    new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
    self.q_table[state][action] = new_value

    # Decay epsilon
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

def act(self, game_state: dict):
    """
    Decide the next action based on the current game state.
    """
    state_features = state_to_features(game_state)
    if state_features is None:
        return 'WAIT'  # Default action if state is invalid

    state = tuple(state_features)  # Convert to tuple for Q-Table key

    # Update Q-Table with the reward from the previous action
    if self.last_state is not None and self.last_action is not None:
        # Reward is zero by default; actual reward is set in train.py
        reward = 0
        update_q_table(self, self.last_state, self.last_action, reward, state)

    # Choose action using ε-greedy policy
    if random.random() < self.epsilon:
        action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    else:
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        # Handle multiple actions with the same max Q-value
        max_actions = [action for action, q in zip(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], q_values) if q == max_q]
        action = random.choice(max_actions)

    # Update last state and action
    self.last_state = state
    self.last_action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'].index(action)

    # Log the chosen action for debugging
    self.logger.debug(f"State: {state}, Action: {action}, Epsilon: {self.epsilon}")

    return action

def callbacks_setup(*args, **kwargs):
    """
    Wrapper function to be called by the framework for setup.
    """
    self = args[0]
    setup(self)

def callbacks_act(*args, **kwargs):
    """
    Wrapper function to be called by the framework to decide the next action.
    """
    self = args[0]
    game_state = args[1]
    return act(self, game_state)
