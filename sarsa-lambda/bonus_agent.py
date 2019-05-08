"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
from tile3 import tiles, IHT
import random
class Agent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.numTilings = 8
        self.alpha = 0.5/self.numTilings
        self.gamma = 1
        self.p = 0
        self.max_size = 2048
        self.iht = IHT(self.max_size)
        self.l = 0.9
        self.actions = [0,1,2]
    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.w =np.random.uniform(-0.001,0,self.max_size)
    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.z =np.zeros(self.max_size)
        action = self.choose_action(state)
        self.last_action = action
        self.last_state = state
        return self.last_action
    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        error = reward
        error = self.update(error)
        action = self.choose_action(state)
        error = self.update1 (state,action,error)
        self.w += self.alpha * error * self.z
        self.z = self.gamma * self.l * self.z
        self.last_action = action
        self.last_state = state
        return self.last_action
    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        error = reward
        error = self.update(error)
        self.w += self.alpha * error * self.z
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
           return self.w,self.iht
        pass
    def update(self,error):
        state = self.last_state
        action = self.last_action
        feature = self.mytiles(state,action)
        for i in feature:
            error = error - self.w[i]
            self.z[i] = 1 
        return error
    def update1(self,state,action,error):
        feature = self.mytiles(state,action)
        for i in feature:
            error += self.gamma * self.w[i]
        return error
    def choose_action(self,state):
        i =np.random.uniform()
        if i > self.p:
            actions = self.actions
            list1 = []
            for action in actions:
                feature = self.mytiles(state,action)
                sum1 = 0
                for i in feature:
                    sum1 += self.w[i]               
                list1.append(sum1) 
            #print(list1.index(max(list1)))              
            return list1.index(max(list1))
        else:
            return random.randint(0,2)
    def mytiles(self,position,action):
        a = [self.numTilings * position[0] /1.7,self.numTilings * position[1] / 0.14]
        return tiles(self.iht,self.numTilings,a,[action])

