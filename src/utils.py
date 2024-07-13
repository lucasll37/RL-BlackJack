HIT = 1
STAND = 0


def reward_engineering(state, action, reward):
    player_counts = state[0]
    dealer_counts = state[1]
    usable_Ace = state[2]

    # if player_counts > 17 and action == STAND:
    #     reward = 0.5

    return reward
