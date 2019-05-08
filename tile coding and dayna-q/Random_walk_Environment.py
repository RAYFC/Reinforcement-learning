
from rl_glue import BaseEnvironment
import numpy as np

class Randon_walk_Environment(BaseEnvironment):
    def __init__(self):
        """Declare environment variables."""
        self.endRightPosition = 1000
        self.endLeftPosition = 0
        self.startPosition = 500
    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.current_state = 500
        return self.current_state
    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        is_terminal = False
        reward = 0
        self.current_state += action
        if self.current_state <= self.endLeftPosition:
            reward = -1
            is_terminal = True
            self.current_state = 0
        if self.current_state >= self.endRightPosition:
            reward = 1
            is_terminal =True
            self.current_state = 1000
        return reward,self.current_state,is_terminal
    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
