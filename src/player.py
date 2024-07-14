from utils import load_agent

datapath =  './models/MonteCarlo.pickle'
# datapath = './models/Q-Learning.pickle'
# datapath = './models/SARSA.pickle'
# datapath = './models/ExpectedSARSA.pickle'

agent = load_agent(datapath)
agent.play(num_episodes=100, render_mode='human', print_results=True)
agent.close()