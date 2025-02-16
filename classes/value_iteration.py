import numpy as np
from math import inf
from classes.environment import Environment
from classes.agent import Agent

class ValueIteration():
    def __init__(self, env : Environment, path = "input_files/value-iteration.txt"):
        self.env = env
        self.policy = None
        self.utilities_matrix = None
        self.get_parameters(path)

    def get_parameters(self, path):
        """Gets gamma and epsilon from the input file"""

        file = open(path, "r")
        data = [ligne.strip() for ligne in file.readlines()]
        self.eps = float(data[-1])
        self.gamma = float(data[-2])
        file.close()

    def apply_algorithm(self, history_file : str = "output_files/log-file_VI.txt"):
        """Finds the best policy using the value iteration algorithm. Calculates the utility matrix of each state and the policy corresponding"""

        # Initialisation
        n_row, n_column = self.env.board.shape
        old_utilities_matrix = np.zeros((n_row, n_column))
        new_utilities_matrix = np.zeros((n_row, n_column))
        policy = np.zeros((n_row, n_column)).tolist()
        actions = ["up", "down", "left", "right"]
        to_symbol = {"up":"^", "down":"v", "left":"<", "right":">"}
        difference = 10000

        # History file
        file = open(history_file,"w")   # Clears a potential previous history in the file
        file = open(history_file, "a")
        file.write("History of the utilities after each iteration of the value iteration algorithm.\nThe empty state utility stays at 0, but is changed at the end of the algorithm. \n\n")
        file.write(f"Gamma : {self.gamma}, threshold : {self.eps}\n\n")
        
        iter = 0
        while difference > self.eps:

            old_utilities_matrix = new_utilities_matrix.copy()
            file.write(f"Iteration {iter} : \n\n")
            np.savetxt(file, old_utilities_matrix, fmt='%8.5f')
            file.write("\n")
            
            for i in range(n_row):
                for j in range(n_column):

                    if self.env.board[i,j] == 3.: # Unreachable
                        continue

                    # Finding the best q-value
                    state = np.array([i,j])
                    max_q_value = -inf
                    best_action = None
                    for action in actions:
                        q_value = 0
                        new_states, proba_transitions = self.env.getStatesProba(state, action)
                        for k, elt in enumerate(new_states):
                            q_value += proba_transitions[k]*(self.env.reward(state)+self.gamma*old_utilities_matrix[elt[0], elt[1]])
                        if q_value > max_q_value:
                            max_q_value = q_value
                            best_action = action

                    new_utilities_matrix[i,j] = max_q_value
                    policy[i][j] = best_action
            difference = np.sum(np.abs(new_utilities_matrix - old_utilities_matrix))
            iter+=1

        # Replacing 0 by None for unreachable states
        for i in range(n_row):
            for j in range(n_column):
                if self.env.board[i,j] == 3.: #unreachable
                     new_utilities_matrix[i,j] = None
                     policy[i][j] = None

        policy = np.matrix(policy)
        file.write("Final utilities :\n\n")
        np.savetxt(file, new_utilities_matrix, fmt='%8.5f')
        file.write("\n\nBest actions per state :\n\n")
        np.savetxt(file, policy, fmt="%6s")
        file.close()
        self.policy = policy
        self.utilities_matrix = new_utilities_matrix

    def apply_policy(self, agent : Agent) -> list:
        """Make the agent go to the green square. Return the list of moves the agent had to do"""
        
        assert (self.policy != None).any()
        actions = []
        
        while self.env.board[agent.position[0], agent.position[1]] != 1 and self.env.board[agent.position[0], agent.position[1]] != 2:
            action = self.policy[agent.position[0], agent.position[1]]
            action_done = agent.doAction(action, self.env.board)
            actions.append([action, action_done])
        
        if self.env.board[agent.position[0], agent.position[1]] == 1:
            print("Gagn\xc3\xa9 !!")
        else:
            print("Perdu...")
        return actions