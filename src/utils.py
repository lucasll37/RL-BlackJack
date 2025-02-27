import cloudpickle

HIT = 1
STAND = 0


def reward_engineering(state, action, reward):
    player_counts = state[0]
    dealer_counts = state[1]
    usable_Ace = state[2]

    # if player_counts > 17 and action == STAND:
    #     reward = 0.5

    return reward


def save_agent(agent, filename, verbose=True):
    with open(filename, "wb") as file:
        cloudpickle.dump(agent, file)

    if verbose:
        print(f"\nAgente salvo em {filename}\n")


def load_agent(filename, verbose=True):
    with open(filename, "rb") as file:
        agent = cloudpickle.load(file)

    if verbose:
        print(f"\n\nAgente carregado de {filename}\n")

    return agent
