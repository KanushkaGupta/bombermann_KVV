import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Initialize shared variables
model = None
target_model = None
optimizer = None
memory = deque(maxlen=MEMORY_SIZE)
steps_done = 0
epsilon = EPSILON_START

def will_be_in_blast(position, game_state, time_step=0):
    """
    Determine if a given position will be in the blast radius at a specific time step.
    
    Args:
        position (tuple): (x, y) coordinates of the position to check.
        game_state (dict): Current game state.
        time_step (int): The time step to consider for bomb explosions.
    
    Returns:
        bool: True if the position is within any blast radius at the given time step, False otherwise.
    """
    x, y = position
    field = game_state['field']
    bombs = game_state['bombs']

    for bomb_pos, countdown in bombs:
        explosion_time = countdown  # Bomb will explode when countdown reaches 0
        steps_until_explosion = explosion_time - time_step
        if steps_until_explosion != 0:
            continue  # Bomb does not explode at this time step

        # Compute the blast zone considering walls
        blast_positions = compute_blast_zone(bomb_pos, field)
        if position in blast_positions:
            return True
    return False

def compute_blast_zone(bomb_pos, field):
    x_bomb, y_bomb = bomb_pos
    blast_positions = [bomb_pos]
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for dx, dy in directions:
        for i in range(1, 4):  # Blast range is 3 tiles
            x, y = x_bomb + i*dx, y_bomb + i*dy
            if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
                break  # Out of bounds
            if field[x, y] == -1:  # Stone wall blocks blast
                break
            blast_positions.append((x, y))
            if field[x, y] == 1:  # Crate blocks blast beyond this point
                break
    return blast_positions



# common.py

# ... [imports and other definitions] ...

def state_to_features(game_state: dict) -> np.ndarray:
    """
    Convert the game state to a feature vector.
    This function should be customized based on your feature engineering.
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

    # 3. Distance to nearest coin
    coins = game_state['coins']
    if coins:
        distances = [np.linalg.norm(np.array([x, y]) - np.array(coin)) for coin in coins]
        min_coin_distance = min(distances)
    else:
        min_coin_distance = 0
    features.append(min_coin_distance)

    # 4. Danger level (e.g., bombs in the vicinity)
    explosion_map = game_state['explosion_map']
    danger_level = explosion_map[x, y]
    features.append(danger_level)

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

    # 6. Distance to nearest opponent
    others = game_state['others']
    if others:
        opponent_positions = [opponent[-1] for opponent in others]
        opponent_distances = [np.linalg.norm(np.array([x, y]) - np.array(pos)) for pos in opponent_positions]
        min_opponent_distance = min(opponent_distances)
    else:
        min_opponent_distance = 0
    features.append(min_opponent_distance)

    # 7. Number of active bombs
    bombs = game_state['bombs']
    num_active_bombs = len(bombs)
    features.append(num_active_bombs)

    # 8. Number of opponents
    num_opponents = len(others)
    features.append(num_opponents)

    # 9. Your current score
    features.append(score)

    # 10. Is the agent currently in danger?
    in_danger = int(will_be_in_blast((x, y), game_state, time_step=0))  # Providing default time_step=0
    features.append(in_danger)

    # 11. Time until the nearest bomb explodes
    if bombs:
        min_bomb_countdown = min([bomb[1] for bomb in bombs])
    else:
        min_bomb_countdown = 0
    features.append(min_bomb_countdown)

    # 12. Number of escape routes
    num_safe_directions = sum(safe_directions)
    features.append(num_safe_directions)

    # Ensure feature vector has consistent length
    while len(features) < 18:
        features.append(0)

    return np.array(features, dtype=float)

def get_new_position(position, action):
    x, y = position
    if action == 'UP':
        return x, y - 1
    elif action == 'DOWN':
        return x, y + 1
    elif action == 'LEFT':
        return x - 1, y
    elif action == 'RIGHT':
        return x + 1, y
    return x, y


