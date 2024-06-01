import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("Blackjack-v1", natural=False, sab=False)


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    observation, info = env.reset()

    while True:
        action = policy(observation)
        states.append(observation)
        actions.append(action)

        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)

        if done:
            break

    return states, actions, rewards


def td_zero_prediction(policy, env, n_episodes, alpha=0.01, gamma=1.0):
    value_table = defaultdict(float)

    for _ in range(n_episodes):
        observation, info = env.reset()
        while True:
            action = policy(observation)
            next_observation, reward, done, truncated, info = env.step(action)

            old_value = value_table[observation]
            next_value = value_table[next_observation] if not done else 0
            value_table[observation] += alpha * (
                reward + gamma * next_value - old_value
            )

            observation = next_observation

            if done:
                break

    return value_table


if __name__ == "__main__":
    value_table = td_zero_prediction(sample_policy, env, n_episodes=50000)

    for state, value in list(value_table.items())[:5]:
        print(f"Estado: {state}, Valor: {value}")
