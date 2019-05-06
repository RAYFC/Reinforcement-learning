import numpy as np
from rl_glue import BaseAgent
import random


class RandomAgent(BaseAgent): 
    def __init__(self):
        self.prevAction = None
        self.alpha=0.1
        self.num_bandits=10
    def agent_init(self):
        """Initialize agent variables."""
        self.prob = 0
        self.state=[5]*self.num_bandits
    def _choose_action(self):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """
        return random.randint(0,self.num_bandits-1)
    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """

        # This agent doesn't care what state it's in, it always chooses
        # to move left or right randomly according to self.probLeft
        self.prevAction = self._choose_action()

        return self.prevAction

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """
        prev_val= self.state[self.prevAction]
        self.state[self.prevAction]=prev_val+self.alpha*(reward-prev_val)
        val=max(self.state)
        index=self.state.index(val)
        self.prevAction=index
        i=random.uniform(0,1)
        if i < 1-self.prob:
            self.prevAction=index
            return index
        else:
            index=random.randint(0,self.num_bandits-1)
            self.prevAction=index

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # random agent doesn't care about reward
        pass

    def agent_message(self, message):
        if 'prob' in message:
            self.prob = float(message.split()[1])
