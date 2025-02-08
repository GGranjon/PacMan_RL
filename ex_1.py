from classes.environment import Environment
from classes.agent import Agent
import numpy as np

env = Environment()
agent = Agent()

def choseRandomMove():
    moves = ["up", "down", "left", "right"]
    probabilites = [0.25, 0.25, 0.25, 0.25]
    return np.random.choice(moves, p=probabilites)

def value_policy_algorithm():
    """Finds the best policy using the value policy algorithm. Returns the utility matrix of each state"""

    n_row, n_column = env.board.shape
    old_utilities_matrix = np.zeros((n_row, n_column))
    new_utilities_matrix = np.zeros((n_row, n_column))
    difference = 10000  # false initialisation to pass the while a first time

    while difference > env.eps:
        print(difference)
        old_utilities_matrix = new_utilities_matrix.copy()
        actions = ["up", "down", "left", "right"]

        for i in range(n_row):
            for j in range(n_column):
                state = np.array([i,j])
                max_q_value = 0
                for action in actions:
                    q_value = 0
                    new_states, proba_transitions = env.getStatesProba(state, action)
                    for k, state in enumerate(new_states):
                        q_value += proba_transitions[k]*(env.reward(state)+env.gamma*old_utilities_matrix[state[0], state[1]])
                    if q_value > max_q_value:
                        max_q_value = q_value

                # at this point, we have the best q_value, meaning the utility of the state
                new_utilities_matrix[i, j] = max_q_value
        difference = np.sum(np.abs(new_utilities_matrix - old_utilities_matrix))
    return new_utilities_matrix

print(value_policy_algorithm())