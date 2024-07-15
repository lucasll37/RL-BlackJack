from utils import load_agent

# datapath =  './models/MonteCarlo.pickle'
# datapath = './models/TemporalDifference.pickle'
# datapath = './models/SARSA.pickle'
# datapath = './models/QLearning.pickle'
datapath = './models/DeepQLearning.pickle'

agent = load_agent(datapath)
agent.play(episodes=50, render_mode='human', print_results=True)
agent.close()
