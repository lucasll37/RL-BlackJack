from utils import load_agent

datapath = "./models/QuasiOptimum_MonteCarlo.pickle"
# datapath = "./models/MonteCarlo.pickle"
# datapath = './models/SARSA.pickle'
# datapath = './models/QLearning.pickle'
# datapath = './models/DeepQLearning.pickle'

agent = load_agent(datapath)
agent.play(episodes=100, render_mode="human", print_results=True)
agent.close()
