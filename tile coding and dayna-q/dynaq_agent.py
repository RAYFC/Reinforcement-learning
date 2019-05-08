
from rl_glue import BaseAgent
import numpy as np
import random

class DynaqAgent(BaseAgent):
    def __init__(self):
        """Declare agent variables."""
        self.actions = [(1,0),(0,1),(0,-1),(-1,0)]
        self.e = 0.1
        self.n = 0
        self.alpha = 0.1
        self.gamma = 0.95
    def agent_init(self):
        self.Q ={}
        self.Models ={}
        for i1 in range(9):
            for i2 in range(6):
                for action in self.actions:
                    self.Q[((i1,i2),action)] = 0
    def agent_start(self, state):
        self.rewards = []
        self.last_action = self.generate_action(state)
        self.last_state = state
        return self.last_action
    def agent_step(self, reward, state):
        action = self.generate_action(state)
        max_action =  self.generate_max_action(state)
        self.Q[(self.last_state,self.last_action)] += self.alpha*(reward + self.gamma*self.Q[(state,max_action)] - self.Q[(self.last_state,self.last_action)])     
        self.Models[(self.last_state,self.last_action)] = (reward,state)
        self.last_state = state 
        self.last_action = action
        self.repeat()
        return action
    def agent_end(self, reward):
        self.Q[(self.last_state,self.last_action)] += self.alpha*(reward - self.Q[(self.last_state,self.last_action)])
    def agent_message(self, in_message):
        if in_message == "n = 0":
            self.n = 0
        elif in_message == "n = 5":
            self.n =5
        elif in_message == "n = 50":
            self.n =50
        else:
            return "error"
    def generate_action(self,state):
        i = np.random.uniform(0,1)
        if i > self.e:
            L=[]
            for action in self.actions:
                L.append(self.Q[(state,action)])
            index=L.index(max(L))
            return self.actions[index]
        else:
            return random.choice(self.actions)
    def generate_max_action(self,state):
        L=[]
        for action in self.actions:
            L.append(self.Q[(state,action)])
        index=L.index(max(L))
        return self.actions[index]

    def repeat(self):
        n = 1
        while n <= self.n:
            n += 1
            (last_state,last_action)=random.choice(list(self.Models.keys()))
            (reward,state) = self.Models[(last_state,last_action)]
            max_action = self.generate_max_action(state)
            self.Q[(last_state,last_action)] += self.alpha*(reward + self.gamma*self.Q[(state,max_action)] - self.Q[(last_state,last_action)])