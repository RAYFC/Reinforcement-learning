from rl_glue import RLGlue
from grid_env import GridEnvironment
from dynaq_agent import DynaqAgent
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_time = time.time()
    max_steps = 8000
    num_runs  = 10
    num_episodes = 50
    # Create and pass agent and environment objects to RLGlue
    environment =GridEnvironment()
    agent = DynaqAgent()
    rlglue = RLGlue(environment, agent)
    rlglue.rl_agent_message('n = 0')
    del agent, environment  # don't use these anymore
    steps1 = {}
    L1 = []
    L2 = [0]*num_episodes
    for episode in range(num_episodes):
      steps1[episode]=[]
      L1.append(episode+1)
    for run in range(num_runs):
      np.random.seed(run)
      rlglue.rl_init()
      step = 0
      for episode in range(num_episodes):
        rlglue.rl_episode(max_steps)
        new_step = rlglue.num_steps()
        steps1[episode].append(new_step-step)
        step = new_step
    for episode in range(num_episodes):
      L2[episode] = sum(steps1[episode])/num_runs
    plt.plot(L1[1:],L2[1:],label = '0 planning steps')

    environment =GridEnvironment()
    agent = DynaqAgent()
    rlglue = RLGlue(environment, agent)
    rlglue.rl_agent_message('n = 5')
    del agent, environment  # don't use these anymore
    steps1 = {}
    L1 = []
    L2 = [0]*num_episodes
    for episode in range(num_episodes):
      steps1[episode]=[]
      L1.append(episode+1)
    for run in range(num_runs):
      np.random.seed(run)
      rlglue.rl_init()
      step = 0
      for episode in range(num_episodes):
        rlglue.rl_episode(max_steps)
        new_step = rlglue.num_steps()
        steps1[episode].append(new_step-step)
        step = new_step
    for episode in range(num_episodes):
      L2[episode] = sum(steps1[episode])/num_runs
    plt.plot(L1[1:],L2[1:],label = '5 planning steps')

    environment =GridEnvironment()
    agent = DynaqAgent()
    rlglue = RLGlue(environment, agent)
    rlglue.rl_agent_message('n = 50')
    del agent, environment  # don't use these anymore
    steps1 = {}
    L1 = []
    L2 = [0]*num_episodes
    for episode in range(num_episodes):
      steps1[episode]=[]
      L1.append(episode+1)
    for run in range(num_runs):
      np.random.seed(run)
      rlglue.rl_init()
      step = 0
      for episode in range(num_episodes):
        rlglue.rl_episode(max_steps)
        new_step = rlglue.num_steps()
        steps1[episode].append(new_step-step)
        step = new_step
    for episode in range(num_episodes):
      L2[episode] = sum(steps1[episode])/num_runs
    plt.plot(L1[1:],L2[1:],label = '50 planning steps')
    plt.xlabel("episodes")
    plt.ylabel("Steps per episode")
    plt.legend()
    plt.show()