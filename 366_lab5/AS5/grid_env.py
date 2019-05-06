from rl_glue import BaseEnvironment
import numpy as np
import random

class GridEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""
        self.blocks = [(2,2),(2,3),(2,4),(5,1),(7,3),(7,4),(7,5)]
    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
    def env_start(self):
        """
        Arguments: Nothing
        Returns: state 
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.state = (0,3)
        return self.state
    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        reward = 0
        terminal = False
        temp = self.state
        self.state =(self.state[0]+action[0],self.state[1]+action[1])
        if self.state in self.blocks:
            self.state = temp
        if self.state[0] > 8:
            self.state=(8,self.state[1])
        if self.state[0] < 0:
            self.state=(0,self.state[1])        
        if self.state[1] > 5:
            self.state=(self.state[0],5)
        if self.state[1] < 0:
            self.state=(self.state[0],0)
        if self.state == (8,5):
            reward = 1
            terminal = True 
        #print (self.state)
        return reward,self.state,terminal    
    def env_message(self, in_message):
        pass
