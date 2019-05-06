"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
import random

class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        
        """
        self.pi={}
        for i in range (1,100):
           self.pi[i] =  min(i, 100-i) # 1 #initialize the policy  
        self.Q =np.zeros((100,51))   #no Values for 0 and 100
        self.times = np.zeros((100,100))   #will not count the # of apperance of state 0 and 100
    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.returns = []   #What (s,a) appears in this episode?
        state = int(state[0])  #transfer the state given to an int
        action= random.randint(1, min(state, 100-state)) #for exploring, random policy
        self.returns.append((state,action)) #Add the first (s,a) to the sequence of episodes
        self.times[state][action] +=1   
        return action
    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        state=int(state[0])
        action = int(self.pi[state])  #select an action according to the state and policy  
        if  (state,action) not in self.returns:
            self.returns.append((state,action))#if it didn't not appear before,add it the the episodes
        return action
    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        for r in self.returns: #pairs(s,a) which appear in this episode
            self.times[r[0]][r[1]] +=1  #the # of appearance + 1
            self.Q[r[0]][r[1]] += (reward-self.Q[r[0]][r[1]])/self.times[r[0]][r[1]] #update the values for all 
        #for i in range(1,100): #for each 
            #m = np.argmax(self.Q[r[0]]) #argmax is tooooooooooooo slow.
            #self-defined function performs much better
            m=self.arg_max(r[0])
            self.pi[r[0]] = m
        #print(self.pi) 
        #print(np.max(self.Q, axis=1))
    def arg_max(self,i):
        l2=[]
        max_value = np.max(self.Q[i])
        for m in range(1,min(i,100-i)+1):
            if self.Q[i][m] == max_value:
                l2.append(m)
        action = random.choice(l2)
        return action
        
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return (np.max(self.Q, axis=1)).tostring()
        else:
            return "I dont know how to respond to this message!!"
