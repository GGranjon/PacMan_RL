from classes.environment import Environment
from classes.agent import Agent
from classes.value_iteration import ValueIteration
from classes.qlearning import QLearning

if __name__ == "__main__":

    # Initialisation
    env = Environment("input_files/Q-Learning.txt")
    agent = Agent()
    value_iteration = ValueIteration(env)

    print(f"initial agent position : {agent.position}\n")

    #value_iteration.apply_algorithm()   # Calculates the policy and utilities
    #actions = value_iteration.apply_policy(agent)   # Applies policy to the agent

    #print(f"path taken by the agent ( [action tried, action done] ) : {actions}")

    qlearning = QLearning(env)
    q_values = qlearning.apply_algorithm(num_iter=15000)

    print(q_values)
    print(qlearning.create_action_matrix(q_values))

