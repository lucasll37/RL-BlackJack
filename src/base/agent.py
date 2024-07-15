import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas import Series
from collections import defaultdict
from abc import ABC
from scipy.optimize import curve_fit
from utils import save_agent


class BJAgent(ABC):

    def __init__(self, render_mode=None, gamma=1, initial_epsilon=1, natural=False, sab=False, max_iteration=21):
        self.env = gym.make("Blackjack-v1", natural=natural, sab=sab, render_mode=render_mode)
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = self.initial_epsilon
        self.natural = natural
        self.sab = sab
        self.max_iteration = max_iteration
        self.map_state_Q = defaultdict(int)
        self.n_actions = self.env.action_space.n
        self.n_states = 0
        self.history = []
        self.validate_each_episodes = None

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


    def learn(self, episodes=10_000, final_epsilon=0.01, epsilon_decay=None, epsilon_val=None, validate_each_episodes=None, verbose=True, save=True):

        raise NotImplementedError('Please implement this method')


    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[self.map_state_Q[state]])
        

    def play(self, episodes=1_000, render_mode="human", print_results=True, epsilon=None):
        env = gym.make("Blackjack-v1", natural=self.natural, sab=self.sab, render_mode=render_mode)
        win = 0
        lose = 0

        if epsilon is None:
            epsilon = self.epsilon

        for i in range(episodes):
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

        coordenate = np.array([self.validate_each_episodes * i for i in range(len(evolution))])
        curve_fit, asymptote = self.fit_exponential(coordenate, evolution)

        ax.set_title(f"[{self.name}] BLACKJACK - WIN RATE EVOLUTION", fontsize=16)
        sns.lineplot(x=coordenate, y=evolution, color="green", label="Win rate", linestyle="-", linewidth=1, ax=ax, alpha=0.4)
        sns.lineplot(x=coordenate, y=curve_fit, color='blue', label=f'Fitted curve (asymptote: {asymptote:.3f})', linestyle='-', linewidth=1.5, ax=ax)
        ax.set_ylabel("Win rate", color="blue", fontsize=14)
        ax.set_xlabel("Episode", color="blue", fontsize=14)
        ax.tick_params(axis="x", colors="blue")
        ax.tick_params(axis="y", colors="blue")
        ax.set_facecolor("#F0FFFF")
        
        plt.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close()

        Series(curve_fit).to_csv(f"./data/{self.name}.csv", index=False)

        if return_fig:
            return fig
        
        
    def fit_exponential(self, x, y):

        def curve(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            params, _ = curve_fit(curve, x, y)
            a, b, c = params        
            y_fitted = curve(x, a, b, c)
            asymptote = a + c
            
        return y_fitted, asymptote
    

    def validation(self, episode, episodes, epsilon_val, verbose, save):

        if not isinstance(epsilon_val, (int, float)):
            _epsilon_val = self.epsilon
        else:
            _epsilon_val = epsilon_val
            
        result = self.play(episodes=1_000, render_mode=None, print_results=False, epsilon=_epsilon_val)
        self.history.append(result)
            
        if verbose:
            print(f"Episode: {episode:7d}/{episodes}, epsilon: {self.epsilon:.5f}, Win: {result[0]}, Lose: {result[1]}, Win rate: {result[0]/(result[0]+result[1]):.3f}")

        if save:
            save_agent(self, f"./models/{self.name}.pickle")


    def epsilon_update(self, episodes, validate_each_episodes, final_epsilon, epsilon_decay):
        self.validate_each_episodes = validate_each_episodes

        if isinstance(epsilon_decay, (int, float)):
            epsilon_decay_factor = epsilon_decay
        else:
            epsilon_decay_factor = np.power(final_epsilon, 1/episodes)

        return epsilon_decay_factor