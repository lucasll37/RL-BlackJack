from base import BJAgent
from utils import reward_engineering
import numpy as np


class BJAgent_MonteCarlo(BJAgent):

    def __init__(self, render_mode=None, gamma=1, initial_epsilon=1, natural=False, sab=False, max_iteration=500):
        super().__init__(render_mode=render_mode, gamma=gamma, initial_epsilon=initial_epsilon, natural=natural, sab=sab, max_iteration=max_iteration)
        self.name = "MonteCarlo"


    def learn(self, iterations=10_000, final_epsilon=0.01, epsilon_decay=None, epsilon_val=None, validate_each_iteration=None, verbose=True, save=True):

        epsilon_decay_factor = self.epsilon_update(iterations, validate_each_iteration, final_epsilon, epsilon_decay)

        for iteration in range(1, iterations+1):
            print(f"Iteration: {iteration:7d}/{iterations}, epsilon: {self.epsilon:.5f}")

            G = 0
            state, _ = self.env.reset()
            done = False
            episodes = []

            while not done:
                action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                reward = reward_engineering(state, action, reward)
                episodes.append((state, action, reward))
                state = next_state

            for t in range(len(episodes) - 1, -1, -1):
                state, action, reward = episodes[t]
                G = self.gamma * G + reward
                self.N[self.map_state_Q[state]] += 1
                old_value = self.Q[self.map_state_Q[state], action]
                new_value = old_value + (1 / self.N[self.map_state_Q[state]]) * (G - old_value)
                self.Q[self.map_state_Q[state], action] = new_value

            if (isinstance(validate_each_iteration, int) and iteration % validate_each_iteration == 0):
                self.validation(iteration, iterations, epsilon_val, verbose, save)
                
            self.epsilon *= epsilon_decay_factor


if __name__ == "__main__":
    agent = BJAgent_MonteCarlo()
    agent.learn(iterations=5_000, final_epsilon=1e-2, validate_each_iteration=5, verbose=True)
    fig = agent.plot_history(return_fig=True)
    fig.savefig(f"./images/{agent.name}.png", dpi=300, format="png")