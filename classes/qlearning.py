from classes.environment import Environment
from classes.agent import Agent
import numpy as np
from random import choice, randint
from json import dump

class QLearning:
    def __init__(self, env : Environment, path = "input_files/Q-Learning.txt"):
        self.env = env
        self.q_values = None
        self.move_to_numeric = {"up":np.array([-1,0]), "down":np.array([1,0]),
                                "left":np.array([0,-1]), "right":np.array([0,1])}
        self.get_parameters(path)

    def get_parameters(self, path):
        """Gets alpha and gamma from the input file"""
        file = open(path, "r")
        data = [ligne.strip() for ligne in file.readlines()]
        self.alpha = float(data[-1])
        self.gamma = float(data[-2])
        file.close()
    
    def apply_algorithm(self, num_iter, history_file : str = "output_files/log-file_QL.txt"):
        """Applies the q-learning algorithm, return a dictionary of all q-values per state discovered"""

        agent = Agent()
        state = str(agent.position[0])+"_"+str(agent.position[1])
        q_values = {}
        q_values[state] = {"up":0, "down":0, "left":0, "right":0}

        # File to save the informations
        file = open(history_file,"w")   # Clears a potential previous history
        file = open(history_file, "a")
        file.write("History of the q-values modifications\n\n")
        file.write(f"ALPHA : {self.alpha}, GAMMA : {self.gamma}, NUMBER OF ITERATIONS : {num_iter}\n\n")

        for i in range(num_iter):

            # Random state switch
            if randint(0,14) == 0:
                rdm_state = choice(list(q_values.keys()))
                rdm_state = np.array([int(rdm_state[0]), int(rdm_state[-1])])
                if (i < 100):
                    file.write(f"Random state switch to {rdm_state}\n\n")
                agent.position = rdm_state
                
            state = str(agent.position[0])+"_"+str(agent.position[1])
            action = max(q_values[state], key=q_values[state].get)

            # Random action switch
            if randint(0,5) == 0:
                action = choice(["up", "down", "left", "right"])
            
            old_pos = agent.position
            chosen_action = agent.doAction(action, self.env.board)
            new_pos = agent.position
            new_state = str(new_pos[0])+"_"+str(new_pos[1])

            # Initialize the new state discovered
            if not new_state in q_values.keys():
                q_values[new_state] = {"up":0, "down":0, "left":0, "right":0}
            
            old_q_value = q_values[state][chosen_action]
            q_values[state][chosen_action] += self.alpha * (self.env.reward(old_pos) + self.gamma * max(q_values[new_state].values()) - q_values[state][chosen_action])
            q_values[state][chosen_action] = round(q_values[state][chosen_action], 6)

            if (i < 100) or (i+1) % 50 == 0:
                file.write(f"Iteration {i+1} : \n\n")
                file.write(f"Etat : {state}, action : {chosen_action}, old q-value : {old_q_value}, new q_value : {q_values[state][chosen_action]}\n\n")

        file.write("\nFINAL Q-VALUES PER STATE : \n\n")
        dump(q_values, file, indent=4)
        file.close()
        self.q_values = q_values

    def create_action_matrix(self, q_values : dict, save_file : str = "output_files/log-file_QL.txt") -> np.matrix:
        """Creates the action matrix based on the q-values dictionary"""

        n,m = self.env.board.shape
        matrice_actions = np.full((n, m), '', dtype=object)
        to_symbol = {"up":"^", "down":"v", "left":"<", "right":">"}

        for etat, actions in q_values.items():
            x, y = map(int, etat.split('_'))
            meilleure_action = max(actions, key=actions.get)
            matrice_actions[x, y] = to_symbol[meilleure_action]

        file = open(save_file, "a")
        file.write("\n\nFINAL ACTIONS TABLE :\n\n")
        np.savetxt(file, matrice_actions, fmt="%6s")
        file.close()
        return matrice_actions

            
            


