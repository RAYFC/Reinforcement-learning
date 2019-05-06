from rl_glue import BaseEnvironment
import numpy as np
import random

class WindyEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""
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
        reward = -1
        terminal = False
        #self.state =(self.state[0]+action[0],self.state[1]+action[1])
        if self.state[0] == 3: 
            self.state=(self.state[0],self.state[1]+1)
        if self.state[0] == 4:
            self.state=(self.state[0],self.state[1]+1)
        if self.state[0] == 5: 
            self.state=(self.state[0],self.state[1]+1)
        if self.state[0] == 8: 
            self.state=(self.state[0],self.state[1]+1)
        if self.state[0] == 6:
            self.state=(self.state[0],self.state[1]+2)
        if self.state[0] == 7:
            self.state=(self.state[0],self.state[1]+2)
        self.state =(self.state[0]+action[0],self.state[1]+action[1])
        if self.state[0] > 9:
            self.state=(9,self.state[1])
        if self.state[0] < 0:
            self.state=(0,self.state[1])        
        if self.state[1] > 6:
            self.state=(self.state[0],6)
        if self.state[1] < 0:
            self.state=(self.state[0],0)
        if self.state == (7,3):
            reward = 0
            terminal = True 
        return reward,self.state,terminal    
    def env_message(self, in_message):
        pass
