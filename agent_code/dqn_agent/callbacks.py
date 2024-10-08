import numpy as np
import torch
import random
from . import common
import events as e
from collections import deque
from .common import will_be_in_blast, compute_blast_zone  

#Monitor past positions to prevent repetitive actions.
previous_positions = []

def setup(self):
    """
    Initialize the agent's models and optimizer.
    This function is called once before the first round starts.
    """
    input_dim = 18  
    output_dim = len(common.ACTIONS)

    # Set up the primary DQN and the target DQN.
    common.model = common.DQN(input_dim, output_dim).to(common.DEVICE)
    common.target_model = common.DQN(input_dim, output_dim).to(common.DEVICE)
    common.target_model.load_state_dict(common.model.state_dict())
    common.target_model.eval()

    # Initialize optimizer
    common.optimizer = torch.optim.Adam(common.model.parameters(), lr=common.LEARNING_RATE)

    # Initialize epsilon
    common.epsilon = common.EPSILON_START

    self.logger.info("DQN Agent setup complete.")

def act(self, game_state: dict) -> str:
    """
    Decide on an action based on the current game state.
    """
    epsilon = common.epsilon

    # Transform the game state into features.
    features = common.state_to_features(game_state)
    if features is None:
        return 'WAIT'  # Default action

    state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(common.DEVICE)

    # Retrieve bomb-related details from the game state.
    agent_x, agent_y = game_state['self'][-1]
    bombs = game_state['bombs']

    if bombs:
        bomb_positions = [bomb[0] for bomb in bombs]
        bomb_countdowns = [bomb[1] for bomb in bombs]
        bomb_distances = [np.linalg.norm(np.array([agent_x, agent_y]) - np.array(bomb_pos)) for bomb_pos in bomb_positions]
        bomb_distance = min(bomb_distances)
        bomb_countdown = bomb_countdowns[bomb_distances.index(bomb_distance)]
    else:
        bomb_distance = float('inf')
        bomb_countdown = float('inf')

    # Gather bomb-related information from the game state.
    if bomb_countdown <= 2 and bomb_distance <= 3:
        action = safe_move(game_state)
        if action:
            self.logger.debug(f"Escaping bomb: Chose safe action {action}")
            return action
        else:
            return 'WAIT'  # If no escape route is available, wait as a backup option.

    # Epsilon-greedy action selection for exploration/exploitation
    if random.random() < epsilon:
        action = random.choice(common.ACTIONS)
        self.logger.debug(f"Exploration: Chose random action {action}")
    else:
        with torch.no_grad():
            q_values = common.model(state_tensor)
            action_idx = q_values.argmax().item()
            action = common.ACTIONS[action_idx]
            self.logger.debug(f"Exploitation: Chose action {action} with Q-value {q_values[0][action_idx].item()}")

    # Conduct a safety check prior to bomb placement.
    if action == 'BOMB' and not is_safe_to_place_bomb(game_state):
        action = 'WAIT'  # Or choose a different safe action
        self.logger.debug("Decided not to place bomb due to safety concerns.")

    return action

def in_blast_radius(game_state):
    agent_position = game_state['self'][-1]
    return will_be_in_blast(agent_position, game_state)

def is_within_blast(agent_position, bomb_position):
    x_a, y_a = agent_position
    x_b, y_b = bomb_position

    #The bomb detonates within a 3-tile radius both horizontally and vertically.
    return (x_a == x_b and abs(y_a - y_b) <= 3) or (y_a == y_b and abs(x_a - x_b) <= 3)

def safe_move(game_state):
    """
    Find the safest move for the agent, prioritizing moves that take it out of danger.
    """
    agent_position = game_state['self'][-1]
    possible_moves = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}

    best_move = None
    max_distance = 0

    for action, move in possible_moves.items():
        new_position = (agent_position[0] + move[0], agent_position[1] + move[1])
        if is_safe_position(new_position, game_state):
            bomb_distances = [np.linalg.norm(np.array(new_position) - np.array(bomb[0])) for bomb in game_state['bombs']]
            nearest_bomb_distance = min(bomb_distances) if bomb_distances else float('inf')

            if nearest_bomb_distance > max_distance:
                max_distance = nearest_bomb_distance
                best_move = action

    return best_move if best_move else None

def is_safe_position(position, game_state):
    x, y = position
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    
    if field[x, y] != 0:  # 0 means free tile, -1 means stone, 1 means crate
        return False

    if explosion_map[x, y] != 0:
        return False  # The position is located within the explosion area.

    # Verify if the position falls within the blast radius of any bomb.
    if will_be_in_blast(position, game_state):
        return False

    return True

def can_agent_escape(agent_position, game_state, max_depth=10):
    field = game_state['field']
    bombs = game_state['bombs']

    visited = set()
    queue = deque()
    queue.append((agent_position, 0))

    max_bomb_timer = max([countdown for _, countdown in bombs] + [4])  # Include the bomb we just placed

    while queue:
        position, step = queue.popleft()
        if step > max_depth:
            continue

        if (position, step) in visited:
            continue
        visited.add((position, step))

        # Determine if the position is safe at this time step.
        if will_be_in_blast(position, game_state, time_step=step):
            continue

        # If we have survived all previous bomb explosions.
        if step > max_bomb_timer:
            return True

        # Investigate neighboring tiles, including remaining in the current position.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:  # Include 'WAIT'
            x, y = position[0] + dx, position[1] + dy
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                continue  # Out of bounds
            if field[x, y] != 0:
                continue  # Not a free tile
            queue.append(((x, y), step + 1))

    return False  # No escape route has been identified.

def is_safe_to_place_bomb(game_state):
    agent_position = game_state['self'][-1]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']

    # Perform a simulation of bomb placement.
    simulated_bombs = bombs.copy() + [(agent_position, 4)]  # Bomb countdown starts at 4
    simulated_game_state = game_state.copy()
    simulated_game_state['bombs'] = simulated_bombs

    # Determine if the agent can escape.
    return can_agent_escape(agent_position, simulated_game_state, max_depth=10)