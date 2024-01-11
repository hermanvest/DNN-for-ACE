# Imports
from config import Config
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent


def main():
    # Initialize Environment
    # Initialize Agent
    policy_network = Policy_Network()
    agent = DEQN_agent(policy_network)
    # Initialize (Replay)Buffer

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


if __name__ == "__main__":
    main()
