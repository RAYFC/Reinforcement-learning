"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
import random

class agent1(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.alpha = 0.5
        self.gamma = 1
        self.p = 0.5
    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.x = np.identity(1001)
        self.w = np.zeros(1001)
    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.current_state = state
        action = self.choose_action(self.current_state)
        self.last_state = self.current_state
        return action
    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.current_state = state
        self.w += self.alpha*(reward+self.gamma*np.dot(self.w,self.x[self.current_state])-np.dot(self.w,self.x[self.last_state]))*self.x[self.last_state]
        action = self.choose_action(self.current_state)
        self.last_state = self.current_state
        return action
    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.w += self.alpha*(reward-np.dot(self.w,self.x[self.last_state]))*self.x[self.last_state]
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return self.w
    def choose_action(self,state):
        i =np.random.uniform()
        if i >self.p:
            return random.randint(1,100)
        else:
            return -random.randint(1,100)
