"""Train GRPO for ACE-DICE.

Usage: python train.py --version 2016 [--iterations N] [--advantage process|outcome] [--seed S] [--tag NAME]
"""

import argparse
import os

import tensorflow as tf
import yaml

from ace_grpo.env import AceEnv
from ace_grpo.grpo import GRPO

HERE = os.path.dirname(os.path.realpath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["2016", "2023"], required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--advantage", choices=["process", "outcome"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="", help="Suffix for the run directory")
    parser.add_argument("--out", default=os.path.join(HERE, "..", "results", "runs"))
    args = parser.parse_args()

    with open(os.path.join(HERE, "configs", f"ace_dice_{args.version}.yaml")) as f:
        config = yaml.safe_load(f)
    if args.iterations:
        config["grpo"]["iterations"] = args.iterations
    if args.advantage:
        config["grpo"]["advantage"] = args.advantage
    if args.seed is not None:
        config["policy"]["seed"] = args.seed

    tf.random.set_seed(config["policy"]["seed"])
    agent = GRPO(AceEnv(config), config)
    agent.train(os.path.join(args.out, f"dice{args.version}{args.tag}"))


if __name__ == "__main__":
    main()
