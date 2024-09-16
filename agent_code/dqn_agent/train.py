# agent_code/dqn_agent/train.py

import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List
import os
from . import common  
import events as e  

def setup_training(self):
    """
    Setup training-specific parameters.
    This function is called once after setup.
    """
    self.logger.info("Training-specific setup completed.")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called after each step during training. Collect training data here.
    """
    # Ensure training is enabled
    if not self.train:
        return

    # Convert game states to features
    state = common.state_to_features(old_game_state)
    next_state = common.state_to_features(new_game_state)

    # Get action index
    try:
        action = common.ACTIONS.index(self_action)
    except ValueError:
        self.logger.warning(f"Invalid action '{self_action}' encountered.")
        return

    # Calculate reward based on events
    reward = reward_from_events(events)

    # Store the experience in memory
    common.memory.append((state, action, reward, next_state))

    # Perform one step of the optimization (on the target network)
    if len(common.memory) >= common.BATCH_SIZE:
        replay(self)

    # Decay epsilon
    if common.epsilon > common.EPSILON_END:
        common.epsilon *= common.EPSILON_DECAY

    # Log if the agent kills itself
    if e.KILLED_SELF in events:
        self.logger.info("Agent killed itself.")
        self.logger.debug(f"Old game state: {old_game_state}")
        self.logger.debug(f"Action taken: {self_action}")
        self.logger.debug(f"Events: {events}")
        self.logger.debug(f"Features: {state}")

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each round during training. Handle final transitions and model saving.
    """
    # Ensure training is enabled
    if not self.train:
        return

    # Convert last game state to features
    state = common.state_to_features(last_game_state)

    # Get action index
    try:
        action = common.ACTIONS.index(last_action)
    except ValueError:
        self.logger.warning(f"Invalid action '{last_action}' encountered at end of round.")
        return

    # Calculate reward based on events
    reward = reward_from_events(events)

    # Store the final transition (no next state)
    common.memory.append((state, action, reward, None))

    # Save the trained model
    save_model()

    # Perform replay if possible
    if len(common.memory) >= common.BATCH_SIZE:
        replay(self)

def replay(self):
    """
    Perform a training step using experiences from memory.
    """
    # Sample a batch of experiences from memory
    batch = random.sample(common.memory, common.BATCH_SIZE)
    states, actions, rewards, next_states = zip(*batch)

    # Convert to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(common.DEVICE)  # Optimized tensor creation
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(common.DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(common.DEVICE)

    # Handle non-final next states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(common.DEVICE)
    if non_final_mask.sum().item() > 0:
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states if s is not None]).to(common.DEVICE)
    else:
        non_final_next_states = torch.empty((0, states.size(1)), dtype=torch.float32).to(common.DEVICE)

    # Compute Q(s_t, a)
    state_action_values = common.model(states).gather(1, actions).squeeze()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(common.BATCH_SIZE, device=common.DEVICE)
    with torch.no_grad():
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = common.target_model(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = rewards + (common.GAMMA * next_state_values)

    # Compute loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    common.optimizer.zero_grad()
    loss.backward()
    common.optimizer.step()

    # Logging
    self.logger.info(f"Training loss: {loss.item()}")

    # Update the target network periodically
    common.steps_done += 1
    if common.steps_done % common.TARGET_UPDATE == 0:
        update_target_model()
        self.logger.info("Target network updated.")

def reward_from_events(events: List[str]) -> float:
    """
    Map game events to rewards.
    """
    game_rewards = {
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -5,
        e.BOMB_DROPPED: 2,
        e.CRATE_DESTROYED: 5,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 15,
        e.GOT_KILLED: -20,
        e.KILLED_SELF: -25,
        
    }

    reward = 0
    for event in events:
        reward += game_rewards.get(event, 0)

    return reward



def save_model(path="dqn_model.pth"):
    """
    Save the current model to the specified path.
    """
    if common.model is not None:
        # Determine the absolute path relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        torch.save(common.model.state_dict(), save_path)
    
    else:
        print("Error: Model is not initialized and cannot be saved.")
