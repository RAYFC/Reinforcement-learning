import numpy as np
from rl_glue import BaseAgent
import random


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
    def agent_init(self):
        """Initialize agent variables."""
        self.state=[0]*self.num_bandits
    def _choose_action(self):
        return random.randint(0,self.num_bandits-1)
    def agent_start(self, state):
        self.prevAction = self._choose_action()
        return self.prevAction
    def agent_step(self, reward, state):
        prev_val= self.state[self.prevAction]
        self.state[self.prevAction]=prev_val+self.alpha*(reward-prev_val)
        val=max(self.state)
        index=self.state.index(val)
        i=random.uniform(0,1)
        if i < 1-self.prob:
            self.prevAction=index
            return index
        else:
            index=self._choose_action()
            self.prevAction=index
            return index

    def agent_end(self, reward):
        pass

    def agent_message(self, message):
        if 'prob' in message:
            self.prob = float(message.split()[1])
