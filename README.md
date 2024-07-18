# Blackjack Agent

This project implements a set of reinforcement learning agents to play Blackjack using the OpenAI Gym library. Each agent implements a different algorithm.

## Description

The project implements agents that can learn to play Blackjack using various reinforcement learning methods, including Monte Carlo, Q-Learning, Sarsa, and Deep Q-Learning. The goal is to train the agent to maximize the win rate in the game of Blackjack.

## Installation

1. Clone the repository:
    ```sh
    git clone git@github.com:lucasll37/RL-BlackJack.git
    cd blackjack-agent
    ```

### A. Manual Installation

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### B. Installation with Makefile

2. Create the local virtual environment and install dependencies:
    ```sh
    make configure
    ```

3. Activate the virtual environment:
    ```sh
    make build
    ```

4. In both cases, make sure the environment is active.

## Usage

1. Run the creation and training of one of the available agent implementations. For example, to train a Monte Carlo agent:
    ```sh
    python src/bj_monte_carlo.py
    ```

2. Run the trained agent to play Blackjack:
    ```sh
    python src/player.py
    ```

## Project Structure

This project structure includes:

```
blackjack-RL/
├── data/
├── doc/
├── images/
│ ├── Monte Carlo.png
│ ├── MonteCarlo.png
│ ├── Q-Learning.png
│ ├── Sarsa.png
│ └── Temporal Difference.png
|
├── models/
│ └── MonteCarlo.pickle
|
├── src/
│ └── base/
│ |   └── agent.py
| |
│ ├── bj_deep_q_learning.py
│ ├── bj_monte_carlo.py
│ ├── bj_q_learning.py
│ ├── bj_sarsa.py
│ ├── compare.py
│ ├── player.py
│ └── utils.py
|
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

- `data/`: Directory containing datas generated during agent training.
- `doc/`: Directory containing the academic report of the work carried out.
- `images/`: Directory containing images generated during agent training.
- `models/`: Directory containing trained models.
- `src/`: Directory containing the project source code.
- `.gitignore`: File listing files and directories to be ignored by Git.
- `Makefile`: File containing automation scripts.
- `README.md`: Project documentation.
- `requirements.txt`: File listing the project dependencies.

## Using the Makefile

The Makefile included in this project provides useful commands for automating common tasks. Below are descriptions and instructions for using each command available in the Makefile.

### Available Commands

#### `clean`
Cleans all generated files in the project, including the Python virtual environment and Python cache files.

**Usage:**
```sh
make clean
```

### `configure`
Creates a virtual environment and installs all dependencies listed in the `requirements.txt` file.

**Usage:**
```sh
make configure
```

### `build`
Activate local virtual environment and adjust code formatting.

**Usage:**
```sh
make build
```

### `help`
Displays help with a description of all available commands in the Makefile.

**Usage:**
```sh
make help
```
