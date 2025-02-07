from classes.environment import Environment
from classes.agent import Agent
import numpy as np

env = Environment()
agent = Agent()

def choseRandomMove():
    moves = ["up", "down", "left", "right"]
    probabilites = [0.25, 0.25, 0.25, 0.25]
    return np.random.choice(moves, p=probabilites)

for i in range(50):
    print(f"move played : {agent.lastMove}, new position : {agent.position}\n")
    move = choseRandomMove()
    agent.doAction(move, env.board)