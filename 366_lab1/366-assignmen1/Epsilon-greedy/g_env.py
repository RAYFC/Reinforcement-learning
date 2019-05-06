import numpy as np

from rl_glue import BaseEnvironment


class OneStateEnvironment(BaseEnvironment):
    """
    Example single-state environment with two actions
    """

    def __init__(self):
        """Declare environment variables."""
        super().__init__()

    def env_init(self):
        """
        Initialize environment variables.
        """
        self.bandits = np.random.normal(0.0,1.0,10)
        self.num_bandits=10       
        self.alpha=0.1
        self.correct_bandit=np.argmax(self.bandits)
    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        return 0

    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        terminal = False
        reward=np.random.normal(self.bandits[action],1.0)
        try:
            return reward, 0, terminal
        except NameError:
            m = "Invalid action specified in One-State Environment's " \
                "env_step: {}"
            print(m.format(action))
            print("Please only return the integers from 0-9 as actions.\n")
            exit(1)

    def env_message(self, message):
        pass
