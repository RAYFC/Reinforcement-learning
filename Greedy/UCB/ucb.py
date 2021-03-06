"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_step() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the final average reward. Each episode is capped at
100 steps (max_steps).
"""
import matplotlib.pyplot as plt
import numpy as np

from u_env import OneStateEnvironment
from u_agent import RandomAgent
from rl_glue import RLGlue


def experiment2(rlg, num_runs, max_steps):
    optimal_times = np.zeros(max_steps)
    for run in range(num_runs):
        np.random.seed(run)
        rlg.rl_init()
        rlg.rl_start()
        environment=rlg._environment
        for i in range(max_steps):
            reward, state, action, is_terminal = rlg.rl_step()
            if action == environment.correct_bandit:
                optimal_times[i]+=1
    newList = [(x * 100 / num_runs) for x in optimal_times]
    return  newList


def main():
    max_steps = 1000  # max number of steps in an episode
    num_runs = 2000
  # number of repetitions of the experiment
    # Create and pass agent and environment objects to RLGlue
    agent = RandomAgent()
    environment = OneStateEnvironment()
    rlglue = RLGlue(environment, agent)
    del agent, environment
    result = experiment2(rlglue, num_runs, max_steps)
    list1=range(1,max_steps+1)
    plt.plot(list1,result)
    plt.yticks(np.arange(0, 100, step=20))
    plt.ylabel("Optimal %")
    plt.xlabel("Steps")
    plt.show()
if __name__ == '__main__':
    main()
