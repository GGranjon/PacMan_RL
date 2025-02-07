import numpy as np

class Environment():
    def __init__(self, path="input_files/value-iteration.txt"):
        self.set_board(path)

    def set_board(self, path):
        file = open(path, "r")
        data = [ligne.strip() for ligne in file.readlines()]
        self.seuil = float(data[-1])
        self.gamma = float(data[-2])
        self.board = []
        for i, line in enumerate(data):
            if i<len(data)-2:
                line = np.array([int(elt) for elt in line.split(",")])
                self.board.append(line)
        self.board = np.matrix(self.board)
    
    def reward(self,x):
        assert x in [0,1,2,3]
        rewards = {0:-0.04, 1:1., 2:-1., 3:0.}
        return rewards[x]