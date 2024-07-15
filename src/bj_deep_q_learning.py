import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from collections import deque
from base import BJAgent
from utils import reward_engineering
import gymnasium as gym


class BJAgent_DeepQLearning(BJAgent):

    def __init__(
        self,
        render_mode=None,
        gamma=1,
        initial_epsilon=1,
        max_iteration=500,
        maxlen_deque=4_096,
    ):
        super().__init__(
            render_mode=render_mode,
            gamma=gamma,
            initial_epsilon=initial_epsilon,
            max_iteration=max_iteration,
        )
        self.name = "DeepQLearning"
        self.model = self._build_model()
        self.replay_buffer = deque(maxlen=maxlen_deque)

    def _build_model(self):
        initializer = RandomUniform(minval=-0.1, maxval=0.1)

        model = Sequential()
        model.add(Input(shape=(self.dimensions_states,)))
        model.add(Dense(16, activation="relu", kernel_initializer=initializer))
        model.add(Dense(8, activation="relu", kernel_initializer=initializer))
        model.add(
            Dense(self.n_actions, activation="linear", kernel_initializer=initializer)
        )

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
            next_action = self.epsilon_greedy_policy(next_state_array, epsilon=0)

            if not done:
                target[0][action] = (
                    reward
                    + self.gamma
                    * self.model.predict(next_state_array, verbose=0)[0][next_action]
                )

            else:
                target[0][action] = reward

            states.append(state_array[0])
            targets.append(target[0])

        _ = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        return

    def learn(
        self,
        episodes=100,
        final_epsilon=0.01,
        epsilon_decay=None,
        epsilon_val=None,
        validate_each_episodes=None,
        verbose=True,
        save=True,
    ):

        epsilon_decay_factor = self.epsilon_update(
            episodes, validate_each_episodes, final_epsilon, epsilon_decay
        )
        done = True

        for episode in range(1, episodes + 1):

            state, _ = self.env.reset()
            done = False
            iteration = 0
            self.last_Q = self.Q.copy()

            while not done and iteration < self.max_iteration:

                iteration += 1

                action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
                next_state, reward, done, truncated, _ = self.env.step(action)

                done = done or truncated
                reward = reward_engineering(state, action, reward)
                state_array = np.array(state, dtype=np.float32).reshape(1, -1)
                next_state_array = np.array(next_state, dtype=np.float32).reshape(1, -1)
                self.replay_buffer.append(
                    (state_array, action, reward, next_state_array, done)
                )
                self.update_policy()
                state = next_state

            if (
                isinstance(validate_each_episodes, int)
                and episode % validate_each_episodes == 0
            ):
                self.validation(episode, episodes, epsilon_val, verbose, save)

            elif verbose:
                print(f"Episode: {episode:7d}/{episodes}, epsilon: {self.epsilon:.5f}")

            self.epsilon *= epsilon_decay_factor


if __name__ == "__main__":

    agent = BJAgent_DeepQLearning()
    agent.learn(
        episodes=5_000,
        final_epsilon=1e-2,
        epsilon_val=0,
        validate_each_episodes=5,
        verbose=True,
    )

    # from utils import load_agent

    # agent = load_agent("./models/DeepQLearning.pickle")

    history = agent.plot_history(return_fig=True)
    policy = agent.plot_policy(return_fig=True)

    history.savefig(f"./images/{agent.name}_history.png", dpi=300, format="png")
    policy.savefig(f"./images/{agent.name}_policy.png", dpi=300, format="png")
