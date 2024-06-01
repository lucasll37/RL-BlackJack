import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Criando o ambiente
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Definição da rede neural
def build_model():
    model = Sequential([
        Dense(128, input_shape=(3,), activation='relu'),  # A entrada espera três valores: score, dealer_score, e usable_ace
        Dense(env.action_space.n, activation='linear')    # Saída para cada ação possível
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Política epsilon-gananciosa para a seleção de ações
def policy(state, model, epsilon):
    """Escolhe ação baseada na política epsilon-gananciosa."""
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    state_array = np.array(state, dtype=np.float32).reshape(1, -1)  # Conversão para float32
    q_values = model.predict(state_array)
    return np.argmax(q_values[0])

# Função para treinar o modelo usando experiência replay
def train_model(model, memory, batch_size=32, discount_factor=0.99):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Correção para transformar uma lista de tuplas em um array NumPy uniformemente
    states = np.array([list(state) for state in states], dtype=np.float32)
    next_states = np.array([list(state) for state in next_states], dtype=np.float32)
    targets = model.predict(states)

    next_q_values = model.predict(next_states)
    next_q_values[dones] = 0  # Se o episódio terminou, o valor futuro é 0

    targets[np.arange(batch_size), actions] = rewards + discount_factor * np.max(next_q_values, axis=1)

    model.fit(states, targets, epochs=1, verbose=0)

# Inicialização
model = build_model()
memory = deque(maxlen=10000)  # Buffer de replay

# Treinamento
episodes = 10000
initial_epsilon = 1.0  # Inicializando epsilon para exploração total
min_epsilon = 0.1      # Mínimo valor de epsilon
epsilon_decay = 0.995  # Fator de decaimento do epsilon

epsilon = initial_epsilon
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        action = policy(state, model, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = next_state[0]

        # Armazena a transição no buffer de replay
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Treina a rede neural
        train_model(model, memory)

    # Reduz o epsilon gradualmente
    epsilon = max(min_epsilon, epsilon_decay * epsilon)  # Decaimento exponencial do epsilon

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
