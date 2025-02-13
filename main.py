from classes.environment import Environment
from classes.agent import Agent
from classes.value_iteration import ValueIteration
from classes.qlearning import QLearning

if __name__ == "__main__":
    
    agent = Agent()

    # QUESTION 1 : VALUE ITERATION

    # Initialisation
    env = Environment("input_files/value-iteration.txt")
    value_iteration = ValueIteration(env)

    value_iteration.apply_algorithm()   # Calculates the policy and utilities
    
    print(f"Value iteration learnt policy : \n\n {value_iteration.policy}\n\n")
    actions = value_iteration.apply_policy(agent)   # Applies policy on the agent
    print(f"Path taken by the agent ( [action tried, action done] ) : \n\n{actions}\n\n")
    
    # QUESTION 2 : Q-LEARNING

    # Initialisation
    env = Environment("input_files/Q-Learning.txt")
    qlearning = QLearning(env)
    
    qlearning.apply_algorithm(num_iter=15000)
    
    q_values = qlearning.q_values
    print(f"Q-learning learnt policy : \n\n{qlearning.create_action_matrix(q_values)}")

