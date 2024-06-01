import gymnasium as gym
import numpy as np
from collections import defaultdict

# Criação do ambiente Blackjack
env = gym.make("Blackjack-v1", natural=False, sab=False)

def sample_policy(state, policy_dict):
    return policy_dict.get(state, 0)  # Retorna a ação baseada na política

def policy_iteration(env, policy_dict, gamma=1.0, theta=0.0001, max_iterations=1000):
    value_table = defaultdict(float)
    is_policy_stable = False
    iteration = 0

    while not is_policy_stable and iteration < max_iterations:
        # Avaliação da política
        while True:
            delta = 0
            for state in policy_dict:
                old_value = value_table[state]
                action = policy_dict[state]
                env.reset()
                env.player, env.dealer, env.usable_ace = state  # Configura o estado no ambiente
                next_state, reward, done, _ = env.step(action)
                next_state = (env.player, env.dealer, env.usable_ace)
                new_value = reward if done else reward + gamma * value_table[next_state]
                value_table[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:
                break

        # Melhoria da política
        is_policy_stable = True
        for state in policy_dict:
            old_action = policy_dict[state]
            # Avaliar ações possíveis
            action_values = []
            for action in range(2):  # Ações possíveis são 0 (parar) e 1 (pedir)
                env.reset()
                env.player, env.dealer, env.usable_ace = state  # Configura o estado no ambiente
                next_state, reward, done, _ = env.step(action)
                next_state = (env.player, env.dealer, env.usable_ace)
                action_value = reward if done else reward + gamma * value_table[next_state]
                action_values.append(action_value)
            best_action = np.argmax(action_values)
            policy_dict[state] = best_action
            if old_action != best_action:
                is_policy_stable = False

        iteration += 1

    return policy_dict, value_table

# Inicializa a política para cada estado possível
policy_dict = {(score, dealer_score, usable_ace): 0 if score >= 20 else 1
               for score in range(4, 22)
               for dealer_score in range(1, 11)
               for usable_ace in [True, False]}

final_policy, value_table = policy_iteration(env, policy_dict)

# Imprime os resultados
for state, value in list(value_table.items())[:5]:
    print(f"Estado: {state}, Valor: {value}")
