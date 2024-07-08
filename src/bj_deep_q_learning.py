import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt

# Criando o ambiente
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Definição da rede neural
def build_model():
    '''
    Constrói e compila o modelo de rede neural.

    Returns:
        model (Sequential): Modelo de rede neural compilado.
    '''
    model = Sequential([
        Input(shape=(3,)),  # A entrada espera três valores: score, dealer_score, e usable_ace
        Dense(128, activation='relu'),
        Dense(env.action_space.n, activation='linear')  # Saída para cada ação possível
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Política epsilon-gananciosa para a seleção de ações
def policy(state, model, epsilon):
    """
    Escolhe ação baseada na política epsilon-gananciosa.
    
    Args:
        state (tuple): Estado atual do ambiente, esperado como uma tupla de três valores.
        model (Sequential): Modelo de rede neural usado para prever os valores de Q.
        epsilon (float): Probabilidade de escolher uma ação aleatória (exploração).

    Returns:
        int: Ação escolhida.
    """
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    
    # Garantir que o estado tenha a forma correta
    if not isinstance(state, (tuple, list)) or len(state) != 3:
        state = (0, 0, 0)
    state_array = np.array(state, dtype=np.float32).reshape(1, 3)
    q_values = model.predict(state_array)
    return np.argmax(q_values[0])

# Função para treinar o modelo usando experiência replay
def train_model(model, memory, batch_size=32, discount_factor=0.99):
    """
    Treina o modelo usando experiência replay.
    
    Args:
        model (Sequential): Modelo de rede neural a ser treinado.
        memory (deque): Buffer de replay contendo as experiências passadas.
        batch_size (int): Tamanho do lote para treinamento.
        discount_factor (float): Fator de desconto para o valor Q futuro.
    """
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    def ensure_state_shape(state):
        if isinstance(state, (list, tuple)) and len(state) == 3:
            return np.array(state, dtype=np.float32).reshape(3)
        else:
            return np.zeros(3, dtype=np.float32)

    states = np.array([ensure_state_shape(state) for state in states])
    next_states = np.array([ensure_state_shape(state) for state in next_states])
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=bool)

    targets = model.predict(states)
    next_q_values = model.predict(next_states)
    next_q_values[dones] = 0  # Se o episódio terminou, o valor futuro é 0

    targets[np.arange(batch_size), actions] = rewards + discount_factor * np.max(next_q_values, axis=1)
    model.fit(states, targets, epochs=1, verbose=0)

# Inicialização
model = build_model()
memory = deque(maxlen=10000)  # Buffer de replay

# Parâmetros de treinamento
episodes = 1000
initial_epsilon = 1.0  # Inicializando epsilon para exploração total
min_epsilon = 0.1      # Mínimo valor de epsilon
epsilon_decay = 0.995  # Fator de decaimento do epsilon

epsilon = initial_epsilon
rewards_list = []  # Lista para armazenar as recompensas de cada episódio

# Treinamento
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = policy(state, model, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = next_state[0]
        total_reward += reward

        # Armazena a transição no buffer de replay
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Treina a rede neural
        train_model(model, memory)

    rewards_list.append(total_reward)
    # Reduz o epsilon gradualmente
    epsilon = max(min_epsilon, epsilon_decay * epsilon)  # Decaimento exponencial do epsilon

    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Epsilon: {epsilon:.4f}, Total Reward: {total_reward}")

# Função para calcular a média móvel
def moving_average(data, window_size):
    """
    Calcula a média móvel dos dados.
    
    Args:
        data (list): Lista de dados.
        window_size (int): Tamanho da janela para calcular a média móvel.

    Returns:
        list: Lista com as médias móveis.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Avaliação visual da performance
window_size = 50
moving_avg_rewards = moving_average(rewards_list, window_size)

plt.figure(figsize=(12, 5))
plt.plot(range(len(moving_avg_rewards)), moving_avg_rewards, label='Média Móvel das Recompensas')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Média')
plt.title(f'Recompensa Média por Episódio (Janela de {window_size} Episódios)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(rewards_list, bins=20, edgecolor='black')
plt.xlabel('Recompensas')
plt.ylabel('Frequência')
plt.title('Distribuição das Recompensas por Episódio')
plt.grid()
plt.show()

# Testando o modelo treinado
for _ in range(10):
    state = env.reset()[0]
    done = False
    while not done:
        action = policy(state, model, epsilon=0)  # Sem exploração durante o teste
        state, reward, done, truncated, info = env.step(action)
        state = state[0]
        if done:
            print(f"Recompensa obtida: {reward}")