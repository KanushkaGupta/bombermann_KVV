import numpy as np
import torch
import random

from . import common  
import events as e  

def setup(self):
    """
    Initialize the agent's models and optimizer.
    This function is called once before the first round starts.
    """
    
    input_dim = 18  
    output_dim = len(common.ACTIONS)

    # Initialize the main DQN and target DQN
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
    # Access epsilon from common
    epsilon = common.epsilon

    # Convert game state to features
    features = common.state_to_features(game_state)
    if features is None:
        return 'WAIT'  # Default action

    state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(common.DEVICE)

    # Epsilon-greedy action selection
    if random.random() < epsilon:
        action = random.choice(common.ACTIONS)
        self.logger.debug(f"Exploration: Chose random action {action}")
    else:
        with torch.no_grad():
            q_values = common.model(state_tensor)
            action_idx = q_values.argmax().item()
            action = common.ACTIONS[action_idx]
            self.logger.debug(f"Exploitation: Chose action {action} with Q-value {q_values[0][action_idx].item()}")

    return action


