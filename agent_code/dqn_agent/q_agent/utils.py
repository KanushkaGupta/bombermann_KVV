
def get_new_position(position, action):
    """
    Given a current position and an action, return the new position.
    Actions: 'UP', 'RIGHT', 'DOWN', 'LEFT'
    """
    x, y = position
    if action == 'UP':
        return x, y - 1
    elif action == 'RIGHT':
        return x + 1, y
    elif action == 'DOWN':
        return x, y + 1
    elif action == 'LEFT':
        return x - 1, y
    else:
        return x, y  # No movement for invalid actions

def will_be_in_blast(position, game_state, time_step=0):
    """
    Determine if the agent is in the blast range in the current or future steps.
    """
    x, y = position
    explosion_map = game_state['explosion_map']

    # Check current explosion
    if explosion_map[x, y] > time_step:
        return True

    # Predict future explosions based on bombs
    for ((bx, by), countdown) in game_state['bombs']:
        bomb_explode_time = countdown - time_step
        if bomb_explode_time >= 0 and bomb_explode_time <= 4:
            # Explosion range: 3 tiles in each direction
            for dx in range(-3, 4):
                nx = bx + dx
                ny = by
                if 0 <= nx < explosion_map.shape[0]:
                    if nx == x and ny == y:
                        return True
            for dy in range(-3, 4):
                nx = bx
                ny = by + dy
                if 0 <= ny < explosion_map.shape[1]:
                    if nx == x and ny == y:
                        return True
    return False
