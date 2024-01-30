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
            np.ndarray: _description_
        """
        raise NotImplementedError
        # episode buffer = []
        # state = initial state of env
        # for i = 1,N do:
        #   prediction = nn(state)
        #   next_state = env.step(prediction)
        #   episode buffer.add([state, next_state])
        #   next_state = state
        # return episode buffer

    def do_learning_pass(self) -> None:
        raise NotImplementedError

    def main_loop(self) -> None:
        raise NotImplementedError
