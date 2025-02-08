from classes.environment import Environment
from classes.agent import Agent
from math import inf
import numpy as np

env = Environment()
agent = Agent()

def get_states_utilities(env : Environment) -> np.matrix:
    """Finds the best policy using the value policy algorithm. Returns the utility matrix of each state"""

    n_row, n_column = env.board.shape
    old_utilities_matrix = np.zeros((n_row, n_column))
    new_utilities_matrix = np.zeros((n_row, n_column))
    best_actions = np.zeros((n_row, n_column)).tolist()
    difference = 10000  # false initialisation to pass the while a first time

    while difference > env.eps:
        old_utilities_matrix = new_utilities_matrix.copy()
        #print(new_utilities_matrix, "\n")
        actions = ["up", "down", "left", "right"]

        for i in range(n_row):
            for j in range(n_column):
                #print(np.matrix(best_actions))
                if env.board[i,j] == 3.: #unreachable
                    continue
                state = np.array([i,j])
                max_q_value = -inf
                best_action = None
                for action in actions:
                    q_value = 0
                    new_states, proba_transitions = env.getStatesProba(state, action)
                    for k, elt in enumerate(new_states):
                        q_value += proba_transitions[k]*(env.reward(state)+env.gamma*old_utilities_matrix[elt[0], elt[1]])
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = action

                # at this point, we have the best q_value, meaning the utility of the state
                new_utilities_matrix[i,j] = max_q_value
                best_actions[i][j] = best_action
        difference = np.sum(np.abs(new_utilities_matrix - old_utilities_matrix))

    for i in range(n_row):
        for j in range(n_column):
            if env.board[i,j] == 3.: #unreachable
                 new_utilities_matrix[i,j] = None
                 best_actions[i][j] = None

    best_actions = np.matrix(best_actions)
    return new_utilities_matrix, best_actions

utilities, actions = get_states_utilities(env)
print(f"utilities : \n{utilities}\n\n actions : \n{actions}\n")

def value_iteration_algorithm(env : Environment, agent : Agent) -> list:
    """finds the best path to the reward using value iteration"""
    pass
