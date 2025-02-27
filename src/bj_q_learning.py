import numpy as np
from base import BJAgent
from utils import reward_engineering


class BJAgent_QLearning(BJAgent):

    def __init__(
        self,
        render_mode=None,
        gamma=1,
        initial_epsilon=1,
        natural=False,
        sab=False,
        max_iteration=500,
    ):
        super().__init__(
            render_mode=render_mode,
            gamma=gamma,
            initial_epsilon=initial_epsilon,
            natural=natural,
            sab=sab,
            max_iteration=max_iteration,
        )
        self.name = "QLearning"

    def learn(
        self,
        episodes=10_000,
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

                next_action = self.epsilon_greedy_policy(
                    next_state, epsilon=0
                )  # greedy policy
                self.N[self.map_state_Q[state]] += 1
                old_value = self.Q[self.map_state_Q[state], action]
                new_value = old_value + (1 / self.N[self.map_state_Q[state]]) * (
                    reward
                    + self.gamma * self.Q[self.map_state_Q[next_state], next_action]
                    - old_value
                )
                self.Q[self.map_state_Q[state], action] = new_value
                state = next_state

            if (
                isinstance(validate_each_episodes, int)
                and episode % validate_each_episodes == 0
            ):
                self.delta = np.linalg.norm(self.Q - self.last_Q)
                self.last_Q = self.Q.copy()
                self.validation(episode, episodes, epsilon_val, verbose, save)

            elif verbose:
                print(f"Episode: {episode:7d}/{episodes}, epsilon: {self.epsilon:.5f}")

            self.epsilon *= epsilon_decay_factor


if __name__ == "__main__":

    agent = BJAgent_QLearning()
    agent.learn(
        episodes=5_000,
        final_epsilon=1e-2,
        epsilon_val=0,
        validate_each_episodes=5,
        verbose=True,
    )

    history = agent.plot_history(return_fig=True)
    policy = agent.plot_policy(return_fig=True)

    history.savefig(f"./images/{agent.name}_history.png", dpi=300, format="png")
    policy.savefig(f"./images/{agent.name}_policy.png", dpi=300, format="png")