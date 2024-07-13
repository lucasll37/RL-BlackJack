import numpy as np
from base import BJAgent
from utils import reward_engineering, save_agent


class BJAgent_QLearning(BJAgent):

    def __init__(self, render_mode=None, gamma=1, initial_epsilon=1):
        super().__init__(render_mode=render_mode, gamma=gamma, initial_epsilon=initial_epsilon)
        self.name = "Q-Learning"


    def learn(self, iterations=10_000, final_epsilon=0.01, epsilon_decay=None, epsilon_val=None, validate_each_iteration=None, verbose=True):
        self.validate_each_iteration = validate_each_iteration
        
        if isinstance(epsilon_decay, (int, float)):
            epsilon_decay_factor = epsilon_decay
        else:
            epsilon_decay_factor = np.power(final_epsilon, 1/iterations)

        done = True

        for iteration in range(1, iterations+1):

            if done:
                state, _ = self.env.reset()
                done = False

            action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            reward = reward_engineering(state, action, reward)
            next_action = self.epsilon_greedy_policy(next_state, epsilon=0) # greedy policy
            self.N[self.map_state_Q[state]] += 1
            old_value = self.Q[self.map_state_Q[state], action]
            new_value = old_value + (1 / self.N[self.map_state_Q[state]]) * (reward + self.gamma * self.Q[self.map_state_Q[next_state], next_action] - old_value)
            self.Q[self.map_state_Q[state], action] = new_value
            state = next_state

            if (isinstance(validate_each_iteration, int) and iteration % validate_each_iteration == 0):
                if not isinstance(epsilon_val, (int, float)):
                    _epsilon_val = self.epsilon
                else:
                    _epsilon_val = epsilon_val
                    
                result = self.play(num_episodes=10_000, render_mode=None, print_results=False, epsilon=_epsilon_val)
                self.history.append(result)

                if verbose:
                    print(f"Iteration: {iteration:10d}, epsilon: {self.epsilon:.5f}, Win: {result[0]}, Lose: {result[1]}, Win rate: {result[0]/(result[0]+result[1]):.3f}")

            self.epsilon *= epsilon_decay_factor

        self.epsilon = 0


if __name__ == "__main__":
    agent = BJAgent_QLearning()
    agent.learn(iterations=2000, final_epsilon=0.05, epsilon_val=0, validate_each_iteration=50, verbose=True)
    fig = agent.plot_history(return_fig=True)
    fig.savefig(f"./images/{agent.name}.png", dpi=300, format="png")
    save_agent(agent, f"./models/{agent.name}.pickle")