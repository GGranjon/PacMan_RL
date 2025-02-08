from classes.environment import Environment
from classes.agent import Agent
from classes.value_iteration import ValueIteration
from math import inf
import numpy as np

if __name__ == "__main__":

    env = Environment()
    agent = Agent()
    value_iteration = ValueIteration(env)

    value_iteration.apply_algorithm()   # Calculates the policy and utilities

    print(f"initial agent position : {agent.position}\n")

    actions = value_iteration.apply_policy(agent)   # Applies policy to the agent

    print(f"path taken by the agent [action tried, action done] : {actions}")