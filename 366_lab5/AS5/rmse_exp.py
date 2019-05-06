"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from Random_walk_Environment import Randon_walk_Environment
from agent1 import agent1
from agent2 import agent2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
if __name__ == "__main__":
    true_value = np.load("TrueValueFunction.npy")
    num_episodes = 2000
    max_steps = 1000
    num_runs = 30
    # Create and pass agent and environment objects to RLGlue
    environment = Randon_walk_Environment()
    agent1 = agent1()
    rlglue = RLGlue(environment, agent1)
    del agent1, environment  # don't use these anymore

    # episodes at which we are interested in the value function
    key_episodes = []
    for i in range (1,num_episodes):
        if i % 10 ==0:
            key_episodes.append(i)

    Rmse_list = {}
    for episode in key_episodes:
        Rmse_list[episode]=[]
    for run in range(num_runs):
        print("run number for agent 1: {}\n".format(run+1))
        np.random.seed(run)
        rlglue.rl_init()
        for episode in range(num_episodes):
            rlglue.rl_episode(max_steps)
            if episode in key_episodes:
                n = 0
                V = rlglue.rl_agent_message('ValueFunction')
                for i in range(1000):
                    n += (true_value[i]-V[i])**2
                Rmse = math.sqrt(n/1000)
                Rmse_list[episode].append(Rmse)
    result1 = []
    for episode in key_episodes:
        sum1 = sum(Rmse_list[episode])
        result1.append(sum1/num_runs)









    environment = Randon_walk_Environment()
    agent2 = agent2()
    rlglue = RLGlue(environment, agent2)
    del agent2, environment  
    Rmse_list = {}
    for episode in key_episodes:
        Rmse_list[episode]=[]
    for run in range(num_runs):
        print("run number for agent 2: {}\n".format(run+1))
        np.random.seed(run)
        rlglue.rl_init()
        for episode in range(num_episodes):
            rlglue.rl_episode(max_steps)
            if episode in key_episodes:
                n = 0
                V = rlglue.rl_agent_message('ValueFunction')
                for i in range(1000):
                    n += (true_value[i]-V[i])**2
                Rmse = math.sqrt(n/1000)
                Rmse_list[episode].append(Rmse)
    result2 = []
    for episode in key_episodes:
        sum1 = sum(Rmse_list[episode])
        result2.append(sum1/num_runs)
    plt.plot(key_episodes,result1,label = 'Tabular encoding ')
    plt.plot(key_episodes,result2,label = 'Tile coding')
    plt.xlabel("Episodes")
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()