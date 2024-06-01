import gymnasium as gym
import numpy as np
from collections import defaultdict

# Criação do ambiente Blackjack
env = gym.make("Blackjack-v1", natural=False, sab=False)

def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def simulate_step(env, state, action):
    """ Simula uma etapa do ambiente a partir de um estado e ação dados, retornando as consequências. """
    env.reset()  # Resetar o ambiente
    env.s = state  # Definir o estado do ambiente diretamente
    next_state, reward, done, info = env.step(action)
    return next_state, reward, done

def value_iteration(env, policy, theta=0.0001, gamma=1.0):
    value_table = defaultdict(float)
    delta = float('inf')

    while delta > theta:
        delta = 0
        # Iterar sobre todos os estados possíveis
        for state in env.observation_space:
            old_value = value_table[state]
            action = policy(state)
            next_state, reward, done = simulate_step(env, state, action)
            new_value = reward if done else reward + gamma * value_table[next_state]
            value_table[state] = new_value
            delta = max(delta, abs(old_value - new_value))

    return value_table

if __name__ == "__main__":
    value_table = value_iteration(env, sample_policy)

    # Exibe os valores estimados para os primeiros cinco estados
    for state, value in list(value_table.items())[:5]:
        print(f"Estado: {state}, Valor: {value}")
