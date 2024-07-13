from model import BJAgent
from utils import reward_engineering
import numpy as np
from collections import defaultdict


class BJAgent_TemporalDifference(BJAgent):

    def __init__(self, render_mode=None, gamma=1):
        super().__init__(render_mode=None, gamma=1)
        self.name = "Temporal Difference"


    def learn(self, iterations=10_000, final_epsilon=0.01, epsilon_decay=None, epsilon_val=None, validate_each_iteration=None, verbose=True):
        self.validate_each_iteration = validate_each_iteration
        
        if isinstance(epsilon_decay, (int, float)):
            epsilon_decay_factor = epsilon_decay
        else:
            epsilon_decay_factor = np.power(final_epsilon, 1/iterations)

        if not isinstance(epsilon_val, (int, float)):
            epsilon_val = self.epsilon

        done = True

        for iteration in range(1, iterations+1):

            if done:
                state, _ = self.env.reset()
                done = False

            action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            reward = reward_engineering(state, action, reward)
            self.N[self.map_state_Q[state]] += 1
            old_value = self.Q[self.map_state_Q[state], action]
            new_value = old_value + (1 / self.N[self.map_state_Q[state]]) * (reward + self.gamma * self.Q[self.map_state_Q[next_state], action] - old_value)
            self.Q[self.map_state_Q[state], action] = new_value
            state = next_state

            if (isinstance(validate_each_iteration, int) and iteration % validate_each_iteration == 0):
                result = self.play(num_episodes=10_000, render_mode=None, print_results=False, epsilon=epsilon_val)
                self.history.append(result)

                if verbose:
                    print(f"Episode: {iteration:10d}, epsilon: {self.epsilon:.5f}, Win: {result[0]}, Lose: {result[1]}, Win rate: {result[0]/(result[0]+result[1]):.3f}")

            self.epsilon *= epsilon_decay_factor