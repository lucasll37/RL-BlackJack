import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
from base import BJAgent
from utils import reward_engineering, save_agent
import gymnasium as gym


class BJAgent_DeepQLearning(BJAgent):

    def __init__(self, render_mode=None, gamma=1, initial_epsilon=1, maxlen_deque=1_000):
        super().__init__(render_mode=render_mode, gamma=gamma, initial_epsilon=initial_epsilon)
        self.name = "DeepQLearning"
        self.model = self._build_model()
        self.replay_buffer = deque(maxlen=maxlen_deque)


    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.dimensions_states,)))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(4, activation="relu"))
        model.add(Dense(self.n_actions, activation="linear"))

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        return model
    
    def epsilon_greedy_policy(self, state, epsilon):   
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        
        else:
            state_array = np.array(state).reshape(1, 3)
            q_values = self.model.predict(state_array, verbose=0)[0]
            
            return np.argmax(q_values)
        

    def update_policy(self, batch_size=32):

        if len(self.replay_buffer) < batch_size:
            return
                
        minibatch = random.sample(self.replay_buffer, batch_size)
        states = list()
        targets = list()

        for state_array, action, reward, next_state_array, done in minibatch:
            target = self.model.predict(state_array, verbose=0)

            if not done:
                target[0][action] = reward + self.gamma * self.model.predict(next_state_array, verbose=0)[0][action]

            else:
                target[0][action] = reward

            states.append(state_array[0])
            targets.append(target[0])

        _ = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        return


    def learn(self, iterations=100, final_epsilon=0.01, epsilon_decay=None, epsilon_val=None, validate_each_iteration=None, verbose=True):
        self.validate_each_iteration = validate_each_iteration

        if isinstance(epsilon_decay, (int, float)):
            epsilon_decay_factor = epsilon_decay
        else:
            epsilon_decay_factor = np.power(final_epsilon, 1/iterations)

        done = True

        for iteration in range(1, iterations+1):
            print(f"iteração: {iteration}/{iterations}")
            if done:
                state, _ = self.env.reset()
                done = False


            action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            reward = reward_engineering(state, action, reward)

            state_array = np.array(state, dtype=np.float32).reshape(1, -1)
            next_state_array = np.array(next_state, dtype=np.float32).reshape(1, -1)

            self.replay_buffer.append((state_array, action, reward, next_state_array, done))
            self.update_policy()
            state = next_state

            if (isinstance(validate_each_iteration, int) and iteration % validate_each_iteration == 0):

                if not isinstance(epsilon_val, (int, float)):
                    _epsilon_val = self.epsilon
                else:
                    _epsilon_val = epsilon_val
                    
                result = self.play(num_episodes=1_000, render_mode=None, print_results=False, epsilon=_epsilon_val)
                self.history.append(result)

                if verbose:
                    print(f"Iteration: {iteration:10d}, epsilon: {self.epsilon:.5f}, Win: {result[0]}, Lose: {result[1]}, Win rate: {result[0]/(result[0]+result[1]):.3f}")

            self.epsilon *= epsilon_decay_factor
            

if __name__ == "__main__":
    agent = BJAgent_DeepQLearning()
    agent.learn(iterations=2_000, final_epsilon=0.05, epsilon_val=0, validate_each_iteration=100, verbose=True)
    fig = agent.plot_history(return_fig=True)
    fig.savefig(f"./images/{agent.name}.png", dpi=300, format="png")
    save_agent(agent, f"./models/{agent.name}.pickle")