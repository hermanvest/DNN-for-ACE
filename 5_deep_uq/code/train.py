"""Train the pseudo-state DEQN for ACE-DICE over the box of uncertain parameters.

Usage: python train.py --version 2016 [--iterations N]
"""

import argparse
import os

import tensorflow as tf
import yaml

from ace_uq.model import AceDiceUQ
from ace_uq.network import build_policy_network
from ace_uq.training import Trainer

HERE = os.path.dirname(os.path.realpath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["2016", "2023"], required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--out", default=os.path.join(HERE, "..", "results", "runs"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    with open(os.path.join(HERE, "configs", f"ace_dice_{args.version}.yaml")) as f:
        config = yaml.safe_load(f)
    if args.iterations:
        config["training"]["iterations"] = args.iterations

    tf.random.set_seed(config["network"]["seed"])
    model = AceDiceUQ(config)
    network = build_policy_network(config)
    trainer = Trainer(model, network, config, os.path.join(args.out, f"dice{args.version}{args.tag}"))
    trainer.train()


if __name__ == "__main__":
    main()
