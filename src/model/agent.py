import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
from collections import defaultdict
from abc import ABC
from scipy.stats import norm


class BJAgent(ABC):

    def __init__(self, render_mode=None, gamma=1, epsilon=1):
        self.env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode=render_mode)
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.epsilon = self.initial_epsilon
        self.map_state_Q = defaultdict(int)
        self.n_actions = self.env.action_space.n
        self.n_states = 0
        self.history = []
        self.validate_each_iteration = None

        _dimensions = defaultdict(float)
        _observation_space = self.env.observation_space
        self.dimensions_states = len(_observation_space)

        for i in range(self.dimensions_states):
            _dimensions[i] = list(range(_observation_space[i].n))

        indexes = [0] * self.dimensions_states

        while True:
            key = list()

            for i in range(self.dimensions_states):
                key.append(_dimensions[i][indexes[i]])

            self.map_state_Q[tuple(key)] = self.n_states
            self.n_states += 1

            indexes[0] += 1

            for i in range(self.dimensions_states - 1):
                if indexes[i] == len(_dimensions[i]):
                    indexes[i] = 0
                    indexes[i + 1] += 1

            if indexes[-1] == len(_dimensions[len(_dimensions) - 1]):
                break

        self.Q = np.random.uniform(low=-1, high=1, size=(self.n_states, self.n_actions))
        self.N = np.zeros(self.n_states)


    def learn(self, iterations=10_000, final_epsilon=0.01, epsilon_decay=None, epsilon_val=None, validate_each_iteration=None, verbose=True):

        raise NotImplementedError('Please implement this method')


    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[self.map_state_Q[state]])
        

    def play(self, num_episodes=1_000, render_mode="human", print_results=True, epsilon=None):
        env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode=render_mode)
        win = 0
        lose = 0

        if epsilon is None:
            epsilon = self.epsilon

        for i in range(num_episodes):
            state, _ = env.reset()
            done = False

            while not done:
                action = self.epsilon_greedy_policy(state, epsilon=epsilon)
                state, reward, done, _, _ = env.step(action)

            if reward == 1:
                win += 1

            else:
                lose += 1

            if print_results:
                print(f"Win: {win}, Lose: {lose}")

        env.close()

        return win, lose

    def close(self):
        self.env.close()

    def reset_learning(self):
        self.Q = np.random.uniform(low=-1, high=1, size=(self.n_states, self.n_actions))
        self.N = np.zeros(self.n_states)
        self.epsilon = self.initial_epsilon
        self.history = []

    def plot_history(self, return_fig=False):
        if len(self.history) == 0:
            raise ValueError("No history to plot")
        
        evolution = []

        for n_win, n_loss in self.history:
            win_rate = n_win / (n_win + n_loss)
            evolution.append(win_rate)

        evolution = np.array(evolution)

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.rcParams["savefig.dpi"] = 300

        coordenate = [self.validate_each_iteration * i for i in range(len(evolution))]
        # rolling_mean, lower_bound, upper_bound = self._calculate_confidence_interval(Series(evolution))
        rolling_mean = Series(evolution).rolling(window=15).mean()

        ax.set_title(f"[{self.name}] Win rate evolution", fontsize=16)
        sns.lineplot(x=coordenate, y=evolution, color="green", label="Win rate", linestyle="-", linewidth="1", ax=ax, alpha=0.4)
        sns.lineplot(x=coordenate, y=rolling_mean, color='blue', label='Moving average', linestyle='-', linewidth=2, ax=ax)
        # ax.fill_between(coordenate, lower_bound, upper_bound, color="lightgreen", alpha=0.3, label='Confidence interval (95%)')
        ax.set_ylabel("Win rate", color="blue", fontsize=14)
        ax.set_xlabel("Episode", color="blue", fontsize=14)
        ax.tick_params(axis="x", colors="blue")
        ax.tick_params(axis="y", colors="blue")
        ax.set_facecolor("#F0FFFF")
        
        plt.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.5)

        if return_fig:
            return fig


    def _calculate_confidence_interval(self, series, window_size=15, confidence_level=0.95):

        alpha = 1 - confidence_level
        z = norm.ppf(1 - alpha / 2)

        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()

        lower_bound = rolling_mean - (z * rolling_std / np.sqrt(window_size))
        upper_bound = rolling_mean + (z * rolling_std / np.sqrt(window_size))

        return rolling_mean, lower_bound, upper_bound
