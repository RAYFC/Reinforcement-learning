
from rl_glue import RLGlue
from windy_env import WindyEnvironment
from n_step_sarsa_agent import SarsaAgent
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_time = time.time()
    max_steps = 8000

    # Create and pass agent and environment objects to RLGlue
    environment =WindyEnvironment()
    agent = SarsaAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore
    rlglue.rl_init()
    L1=[]
    L2=[]
    n = rlglue.rl_agent_message('n')
    a =  rlglue.rl_agent_message('a')
    while rlglue.num_steps() <  max_steps:
      L1.append(rlglue.num_steps())
      rlglue.rl_episode(10000)
      episodes = rlglue.num_episodes()
      L2.append(episodes)
    plt.title(str(n) + '-step sarsa with '+ str(a) +" actions")
    plt.plot(L1,L2)
    plt.show()