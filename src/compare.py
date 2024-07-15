import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


monteCarlo = pd.read_csv("./data/MonteCarlo.csv")
SARSA = pd.read_csv("./data/SARSA.csv")
QLearning = pd.read_csv("./data/QLearning.csv")
DeepQLearning = pd.read_csv("./data/DeepQLearning.csv")

fig, ax = plt.subplots(figsize=(12, 8))
plt.rcParams["savefig.dpi"] = 300

coordenate = [5 * i for i in range(len(monteCarlo))]
_coordenate = [5 * i for i in range(len(DeepQLearning))]


ax.set_title(f"BLACKJACK - WIN RATE EVOLUTION COMPARISON", fontsize=16)

sns.lineplot(
    x=coordenate,
    y=monteCarlo.iloc[:, 0],
    color="green",
    label="Monte Carlo",
    linestyle="-",
    linewidth=1.5,
    ax=ax,
)

sns.lineplot(
    x=coordenate,
    y=SARSA.iloc[:, 0],
    color="orange",
    label="SARSA",
    linestyle="-",
    linewidth=1.5,
    ax=ax,
)

sns.lineplot(
    x=coordenate,
    y=QLearning.iloc[:, 0],
    color="black",
    label="Q-Learning",
    linestyle="-",
    linewidth=1.5,
    ax=ax,
)

sns.lineplot(
    x=_coordenate,
    y=DeepQLearning.iloc[:, 0],
    color="blue",
    label="Deep Q-Learning",
    linestyle="-",
    linewidth=1.5,
    ax=ax,
)

ax.set_ylabel("Win rate", color="blue", fontsize=14)
ax.set_xlabel("Episode", color="blue", fontsize=14)
ax.tick_params(axis="x", colors="blue")
ax.tick_params(axis="y", colors="blue")
ax.set_facecolor("#F0FFFF")

plt.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"./images/comparison.png", dpi=300, format="png")
plt.show()
plt.close()
