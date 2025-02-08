import numpy as np

class Agent():
    def __init__(self, init_position=np.array([2,0])):
        self.position = init_position
        self.move_to_numeric = {"up":np.array([-1,0]), "down":np.array([1,0]),
                                "left":np.array([0,-1]), "right":np.array([0,1])}
        
    def choseRandomAction(self, main_move, alt_move1, alt_move2, probabilities=[0.8, 0.1, 0.1]):
        """Choses a random action between 3 actions given the probabilities of each action"""

        actions = [main_move, alt_move1, alt_move2]
        random_choice = np.random.choice([0,1,2], p=probabilities)
        return actions[random_choice]
    
    def hitWall(self, action: str, board : np.matrix) -> bool:
        """Checks if doing a move in the current position makes the agent hit a wall"""

        new_position = self.position + self.move_to_numeric[action]
        n_row, n_column = board.shape
        if ((new_position[0] < 0) or (new_position[0] >= n_row) or
            (new_position[1] < 0) or (new_position[1] >= n_column)):
            
            # we are outside the board, so we hit a wall
            return True
        
        elif board[new_position[0], new_position[1]] == 3:

            # we hit the wall on the board
            return True
        
        return False

    def doAction(self, move : str, board : np.matrix):
        """Modifies the position the agents is once he plays a move.
        Returns the action the agent ended up doing"""
        
        if move == "up" or move == "down":
            alt_move1 = "left"
            alt_move2 = "right"
        else:
            alt_move1 = "up"
            alt_move2 = "down"
        
        chosen_move = self.choseRandomAction(move, alt_move1, alt_move2)

        if not self.hitWall(chosen_move, board):
            self.position += self.move_to_numeric[chosen_move]
        return chosen_move

