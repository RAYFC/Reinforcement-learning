
from rl_glue import BaseAgent
import numpy as np
import random

class SarsaAgent(BaseAgent):
    def __init__(self):
        """Declare agent variables."""
        self.actions = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        self.chocie = input("How many actions do you want? (Please input 4,8 or 9,default will be 8):")
        try:
            int(self.chocie)
        except:
            pass
        else:
            if  int(self.chocie) == 4:
                self.actions = [(1,0),(0,1),(0,-1),(-1,0)]
            if  int(self.chocie) == 9:
                self.actions = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,0)]
        n =  input("Please input the # of steps,if enter is not valid,default is 1 ie. n (n>=1):")  
        try:
            int(n)
        except:
            self.n = 1
        else:
            self.n = int(n)
        if self.n < 1:
            self.n =1
        self.alpha = 0.5
        self.e = 0.1

    def agent_init(self):
        self.Q ={}
        for i1 in range(10):
            for i2 in range(7):
                for action in self.actions:
                    self.Q[((i1,i2),action)] = 0
    def agent_start(self, state):
        self.last_action = []
        self.last_state  = []
        self.rewards = []
        action = self.generate_action(state)
        self.last_action.append(action)
        self.last_state.append(state)
        return action


    def agent_step(self, reward, state):
        self.rewards.append(reward)
        action = self.generate_action(state)
        if len(self.last_action) == self.n:
            rewards = sum(self.rewards)
            self.Q[(self.last_state[0],self.last_action[0])] += self.alpha*(rewards + self.Q[(state,action)] - self.Q[(self.last_state[0],self.last_action[0])])
            self.last_action.pop(0)
            self.last_state.pop(0)
            self.rewards.pop(0)            
        self.last_state.append(state)
        self.last_action.append(action)
        return action



    def agent_end(self, reward):
        self.rewards.append(reward)
        for i in range(len(self.last_action)):
            rewards = sum(self.rewards[i:])
            self.Q[(self.last_state[i],self.last_action[i])] += self.alpha*(rewards - self.Q[(self.last_state[i],self.last_action[i])])




    def agent_message(self, in_message):
        if in_message == "n":
            return self.n
        if in_message == "a":
            return len(self.actions)
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
