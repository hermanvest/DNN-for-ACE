"""
A class that can take an environment and an agent of some sort as input. Should be generic, so interfaces or some abstract class.
- Do tests to check that the setup is valid
    - Check that the state space of the environment is compatible with the agent.
    - Etc...
- 
"""

import numpy as np
from agents.deqn_agent import DEQN_agent
from environments.abstract_environment import Abstract_Environment


class Algorithm_DEQN:
    def __init__(self, env: Abstract_Environment, agent: DEQN_agent) -> None:
        self.env = env
        self.agent = agent

    def check_compatibility(self) -> None:
        """
        Chekcs if the agent's inputspace is the same as the state space of the environment.
        checks if the agent's outputspace is compatible with the actionspace in the environment.

        Raises:
            ValueError: inputs are not compatible
        """
        # raise ValueError(f"Incompatible parameters: {param1} and {param2}")
        raise NotImplementedError

    def generate_episode(self) -> np.ndarray:
        """
        Generates an episode with the policy network and returns a numpy array with state transitions.
        Args:
            None

        Returns:
            np.ndarray: Array with state transitions as elements [state_t, state_t_plus_1]
        """
        # Returns tensor with [timestep, simulation or batch, statevariables]
        raise NotImplementedError

    def epoch(self):
        raise NotImplementedError
        # 1. divide into batches
        # 2. for batch in batches, calculate errors, do gradient descent
        # 3. return losses

    def do_epochs(self) -> None:
        raise NotImplementedError
        # for epoch in epochs, do epochs thing, log losses

    def main_loop(self) -> None:
        raise NotImplementedError
        # for episode in episodes,
        #   generate episode
        #   run do_epochs on episode
        #   do checkpoint of model if it is performing better
