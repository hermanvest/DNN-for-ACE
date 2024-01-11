"""
Configuration settings for the whole project.

This module contains configuration parameters and hyperparameters for the 
reinforcement learning model and training process.

Attrributes:
    STATE_SPACE (int): 
    ACTION_SPACE (int): 
    TOTAL_EPISODES (int): 
    CHECKPOINT_DIR (str): 
"""


class Config:
    # State and action space
    STATE_SPACE = 10
    ACTION_SPACE = 10

    # Network parameters
    HIDDEN_NODES = 1024

    # Training parameters
    TOTAL_EPISODES = 1000

    # checkpoint parameters
    CHECKPOINT_DIR = "./checkpoints/"
    CHECKPOINT_FREQ = 100
