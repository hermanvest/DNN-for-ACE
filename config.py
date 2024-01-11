"""
Configuration settings for the whole project.

This module contains configuration parameters and hyperparameters for the 
reinforcement learning model and training process.

Attrributes:
    TOTAL_EPISODES (int): 
    CHECKPOINT_DIR (str): 
"""


class Config:
    # Training parameters
    TOTAL_EPISODES = 1000

    # checkpoint parameters
    CHECKPOINT_DIR = "./checkpoints/"
    CHECKPOINT_FREQ = 100
