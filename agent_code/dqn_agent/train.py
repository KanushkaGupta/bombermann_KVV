import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List
import os
from . import common
import events as e
from .common import compute_blast_zone  

previous_positions = []  # Monitor past positions to penalize repeated actions.

def setup_training(self):
    self.logger.info("Training-specific setup completed.")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    current_position = new_game_state['self'][-1]

    # Impose penalties for repetitive behavior.
    if len(previous_positions) > 5 and current_position in previous_positions[-5:]:
        events.append('REPEATED_ACTION')

    # Refresh the position history.
    previous_positions.append(current_position)

    reward = reward_from_events(events, new_game_state)

    # Transform game states into features.
    state = common.state_to_features(old_game_state)
    next_state = common.state_to_features(new_game_state)

    action = common.ACTIONS.index(self_action)

    # Save the experience.
    common.memory.append((state, action, reward, next_state))

    if len(common.memory) >= common.BATCH_SIZE:
        replay(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if not self.train:
        return

    state = common.state_to_features(last_game_state)

    try:
        action = common.ACTIONS.index(last_action)
    except ValueError:
        self.logger.warning(f"Invalid action '{last_action}' encountered at end of round.")
        return

    reward = reward_from_events(events, last_game_state)

    common.memory.append((state, action, reward, None))

    save_model()

    if len(common.memory) >= common.BATCH_SIZE:
        replay(self)

def reward_from_events(events: List[str], game_state) -> float:
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -5,
        e.INVALID_ACTION: -20,
        e.BOMB_DROPPED: 0,
        e.CRATE_DESTROYED: 10,
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 100,
        e.GOT_KILLED: -200,
        e.KILLED_SELF: -500,
        'ESCAPED_BOMB': 30,
        'REPEATED_ACTION': -5,
    }

    reward = 0
    for event in events:
        reward += game_rewards.get(event, 0)

    # Impose penalties for being near bombs that are about to detonate.
    agent_x, agent_y = game_state['self'][-1]
    bombs = game_state['bombs']
    field = game_state['field']

    if bombs:
        for bomb_pos, countdown in bombs:
            if countdown <= 2:
                blast_positions = compute_blast_zone(bomb_pos, field)
                if (agent_x, agent_y) in blast_positions:
                    reward -= 50  # Significant penalty for being in danger

    return reward

def save_model(path="dqn_model.pth"):
    if common.model is not None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(common.model.state_dict(), save_path)
    else:
        print("Error: Model is not initialized and cannot be saved.")

def update_target_model():
    common.target_model.load_state_dict(common.model.state_dict())
    common.target_model.eval()

def replay(self):
    batch = random.sample(common.memory, common.BATCH_SIZE)
    states, actions, rewards, next_states = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(common.DEVICE)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(common.DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(common.DEVICE)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(common.DEVICE)
    if non_final_mask.sum().item() > 0:
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states if s is not None]).to(common.DEVICE)
    else:
        non_final_next_states = torch.empty((0, states.size(1)), dtype=torch.float32).to(common.DEVICE)

    state_action_values = common.model(states).gather(1, actions).squeeze()

    next_state_values = torch.zeros(common.BATCH_SIZE, device=common.DEVICE)
    with torch.no_grad():
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = common.target_model(non_final_next_states).max(1)[0]

    expected_state_action_values = rewards + (common.GAMMA * next_state_values)

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    common.optimizer.zero_grad()
    loss.backward()
    common.optimizer.step()

    self.logger.info(f"Training loss: {loss.item()}")

    common.steps_done += 1
    if common.steps_done % common.TARGET_UPDATE == 0:
        update_target_model()
        self.logger.info("Target network updated.")
        