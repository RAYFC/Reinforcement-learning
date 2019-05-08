#!/usr/bin/env python

import numpy as np
from bonus_agent import Agent
import statistics
from rl_glue import RLGlue
from env_hw6 import Environment
import math

def question_4():
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 50
    max_eps_steps = 1000000

    steps = np.zeros([num_runs, num_episodes])
    rewards = []
    for r in range(num_runs):
        print("run number : ", r+1)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
        reward = rlglue.total_reward()
        rewards.append(reward)
    mean = sum(rewards)/len(rewards)
    stder = statistics.stdev(rewards)/math.sqrt(len(rewards))
    print("mean:",mean)
    print("std:",stder)
    np.save('bonus_steps', steps)
    np.save("mean",mean)
    np.save("stder",stder)

if __name__ == "__main__":
    question_4()
    print("Done")