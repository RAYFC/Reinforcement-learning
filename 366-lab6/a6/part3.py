#!/usr/bin/env python

import numpy as np
from agent_hw6 import Agent
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rl_glue import RLGlue
from env_hw6 import Environment
from tile3 import tiles

def question_3():
    # Specify hyper-parameters
    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)
    num_episodes = 1000
    num_runs = 1
    max_eps_steps = 1000000
    for _ in range(num_runs):
        rlglue.rl_init()
        i = 0
        for i in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            print(i)
    fout = open('value', 'w') 
    steps = 50
    w, iht = rlglue.rl_agent_message("ValueFunction")
    Q = np.zeros([steps, steps])
    for i in range(steps):
        for j in range(steps):
            values = []
            for a in range(3):
                value = 0
                for index in tiles(iht, 8, [8*(-1.2 + (i * 1.7 / steps))/1.7, 8*(-0.07 + (j * 0.14 / steps))/0.14], [a]):
                    value -= w[index]
                values.append(value)
            height = max(values)
            fout.write(repr(height) + ' ') 
            Q[j][i] = height
        fout.write('\n') 
    fout.close() 
    np.save("value",Q)
if __name__ == "__main__":
    question_3()
