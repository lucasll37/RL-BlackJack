import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("Blackjack-v1", natural=False, sab=False)


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    # A política decide "pegar" se o score é menor que 20, e "não pegar" se for 20 ou mais
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


def sarsa(policy, env, n_episodes, alpha=0.01, gamma=1.0):
    # Inicializa a tabela Q com zeros
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(n_episodes):
        observation, info = env.reset()
        action = policy(observation)
        while True:
            next_observation, reward, done, truncated, info = env.step(action)
            next_action = policy(
                next_observation
            )  # Escolhe a próxima ação baseada na política

            # Fórmula de atualização SARSA
            Q[observation][action] += alpha * (
                reward
                + gamma * Q[next_observation][next_action]
                - Q[observation][action]
            )

            observation, action = next_observation, next_action

            if done:
                break

    return Q


if __name__ == "__main__":
    Q = sarsa(sample_policy, env, n_episodes=50000)

    # Imprime alguns valores Q para observar o aprendizado
    for state, actions in list(Q.items())[:5]:
        print(f"Estado: {state}, Ações: {actions}")
