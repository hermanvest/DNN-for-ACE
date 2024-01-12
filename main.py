# Imports
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent

from utils.config_loader import load_config


def main():
    # Initialize Environment
    # Initialize Agent
    network_config = load_config("./configs/network_configs/network_config1.yaml")
    policy_network = Policy_Network(network_config)
    agent = DEQN_agent(policy_network)
    # Initialize (Replay)Buffer

    """
    # Start the main training loop
    for episode in range(Config.TOTAL_EPISODES):
        # reset environment

        done = False
        # start to loop through the episode
        while not done:
            done = True  # REMOVE THIS LATER

            # sample action from agent in initial state
            # take action in environment
            # store observation in buffer

            if (episode % Config.CHECKPOINT_FREQ) == 0:
                # save model
                pass

        # Train Agent? At least by Monte Carlo.
        pass
    pass
    """


if __name__ == "__main__":
    main()
