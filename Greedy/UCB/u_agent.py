import numpy as np
from rl_glue import BaseAgent
import random
import math 

class RandomAgent(BaseAgent): 
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        self.prevAction = None
        self.alpha=0.1
        self.num_bandits=10
        self.prob = 0.1
        self.init=5
        self.c=0.2
    def agent_init(self):
        """Initialize agent variables."""
        self.state=[self.init]*self.num_bandits
        self.times=[1]*self.num_bandits
        self.ucbs=[0]*self.num_bandits
    def _choose_action(self):
        return random.randint(0,self.num_bandits-1)
    def agent_start(self, state):
        self.prevAction = self._choose_action()
        return self.prevAction
    def agent_step(self, reward, state):
        prev_val= self.state[self.prevAction]
        self.state[self.prevAction]=prev_val+self.alpha*(reward-prev_val)
        for i in range(self.num_bandits):
            l=(sum(self.times))/self.times[i]
            self.ucbs[i]=self.state[i]+self.c*(math.sqrt(math.log(l)))
        val=max(self.ucbs)
        index=self.ucbs.index(val)
        m=random.uniform(0,1)
        if m < 1-self.prob:
            self.prevAction=index
            self.times[index]+=1
        else:
            index=self._choose_action()
            self.prevAction=index
            self.times[index]+=1
        return index

    def agent_end(self, reward):
        pass

    def agent_message(self, message):
        if 'prob' in message:
            self.prob = float(message.split()[1])
