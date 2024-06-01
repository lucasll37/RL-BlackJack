import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("Blackjack-v1", natural=False, sab=False)


def sample_policy(Q, observation):
    """Retorna a ação com o maior valor Q para uma observação, escolhendo aleatoriamente se houver empate."""
    return np.argmax(Q[observation])


def q_learning(env, n_episodes, alpha=0.01, gamma=1.0):
    # Inicializa a tabela Q com zeros para cada par estado-ação
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(n_episodes):
        observation, info = env.reset()
        while True:
            # Escolhe a ação com base na política derivada de Q (política gananciosa)
            action = sample_policy(Q, observation)
            next_observation, reward, done, truncated, info = env.step(action)

            # Seleciona a ação que maximiza Q para o próximo estado
            best_next_action = np.argmax(Q[next_observation])

            # Fórmula de atualização do Q-Learning
            Q[observation][action] += alpha * (
                reward
                + gamma * Q[next_observation][best_next_action]
                - Q[observation][action]
            )

            observation = next_observation

            if done:
                break

    return Q


if __name__ == "__main__":
    Q = q_learning(env, n_episodes=50000)

    # Imprime alguns valores Q para observar o aprendizado
    for state, actions in list(Q.items())[:5]:
        print(f"Estado: {state}, Ações: {actions}")
