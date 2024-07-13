from utils import load_agent


agent = load_agent('./models/MonteCarlo.pickle')
agent.play(num_episodes=100, render_mode='human', print_results=True)
agent.close()