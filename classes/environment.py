import numpy as np

class Environment():
    def __init__(self, path="input_files/Q-Learning.txt"):
        self.set_board(path)

    def set_board(self, path):
        file = open(path, "r")
        data = [ligne.strip() for ligne in file.readlines()]
        self.board = []
        for i, line in enumerate(data):
            if i<len(data)-2:
                line = np.array([int(elt) for elt in line.split(",")])
                self.board.append(line)
        self.board = np.matrix(self.board)
        file.close()
    
    def reward(self, state : np.ndarray):
        """returns the reward associated to a state"""

        rewards = {0:-0.04, 1:1., 2:-100, 3:None}
        return rewards[self.board[state[0], state[1]]]
    
    def hitWall(self, state, action: np.ndarray) -> bool:
        """Checks if doing a move in the current state makes the agent hit a wall"""

        new_state = state + action
        n_row, n_column = self.board.shape
        if ((new_state[0] < 0) or (new_state[0] >= n_row) or
            (new_state[1] < 0) or (new_state[1] >= n_column)):
            
            # we are outside the board, so we hit a wall
            return True
        
        elif self.board[new_state[0], new_state[1]] == 3:

            # we hit the wall on the board
            return True
        
        return False
    
    def getStatesProba(self, state : np.ndarray, action : str):
        """returns the new states and probability of each transition after an action at a given state"""

        move_to_numeric = {"up":np.array([-1,0]), "down":np.array([1,0]),
                                "left":np.array([0,-1]), "right":np.array([0,1])}
        main_move = move_to_numeric[action]

        if action == "up" or action == "down":
            alt_move1 = np.array([0,-1])  # left
            alt_move2 = np.array([0, 1])  # right
        else:
            alt_move1 = np.array([-1,0])  # up
            alt_move2 = [1, 0]  # down
        
        actions = [main_move, alt_move1, alt_move2]
        new_states = []
        proba_transitions = [0.8, 0.1, 0.1]
        for elt in actions:
            if self.hitWall(state, elt):
                new_states.append(state)
            else:
                new_states.append(state + elt)
        return new_states, proba_transitions