import numpy as np
import tensorflow as tf
from pathlib import Path

from environments.ace_dice.equations_of_motion_ace_dice_2016 import (
    Equations_of_motion_Ace_Dice_2016,
)
from environments.ace_dice.compute_loss_ace_dice_2016 import Computeloss_Ace_Dice_2016

from utils.config_loader import load_config

# Define the path to the current script
current_script_path = Path(__file__).parent
# Navigate up to the common root and then to the YAML file
yaml_file_path = (
    current_script_path.parent.parent.parent
    / "configs"
    / "state_and_action_space"
    / "ace_dice_2016.yaml"
)

t_max = 10


##################### INITIALIZATION #####################
def get_equations() -> Equations_of_motion_Ace_Dice_2016:
    configs = load_config(yaml_file_path)
    states = configs["state_variables"]
    actions = configs["action_variables"]
    parameters = configs["parameters"]
    return Equations_of_motion_Ace_Dice_2016(t_max, states, actions, parameters)


def get_loss_class() -> Computeloss_Ace_Dice_2016:
    configs = load_config(yaml_file_path)
    parameters = configs["parameters"]
    return Computeloss_Ace_Dice_2016(parameters, get_equations())


#####################  CALCULATION TESTS #####################
def test_ell1_calculation():
    print("\n================== RUNNING: test_ell1_calculation() ==================")
    calc = get_loss_class()

    print("Creating constants...")
    lambda_k_t = tf.constant(1.0, dtype=tf.float32)

    print("Calculating ell1...")
    result = calc.ell_1(lambda_k_t)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell1_calculation() ==================")


def test_ell2_4_calculation():
    print("\n================== RUNNING: test_ell2_4_calculation() ==================")
    calc = get_loss_class()

    print("Creating constants...")
    lambda_m_t = tf.ones([3], dtype=tf.float32)
    lambda_tau_1_t = tf.constant(1.0, dtype=tf.float32)

    print("Calculating ell2_4...")
    result = calc.ell_2_4(lambda_m_t, lambda_tau_1_t)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell2_4_calculation() ==================")


def test_ell5_6_calculation():
    print("\n================== RUNNING: test_ell5_6_calculation() ==================")
    calc = get_loss_class()
    lambda_tau_t_vector = tf.ones([2], dtype=tf.float32)
    result = calc.ell_5_6(lambda_tau_t_vector)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell5_6_calculation() ==================")


def test_ell7_calculation():
    print("\n================== RUNNING: test_ell7_calculation() ==================")
    calc = get_loss_class()
    vars = {
        "k_tplus": tf.constant(1.0, dtype=tf.float32),
        "x_t": tf.constant(0.6, dtype=tf.float32),
        "E_t": tf.constant(1.0, dtype=tf.float32),
        "k_t": tf.constant(1.0, dtype=tf.float32),
        "tau_1_t": tf.constant(1.0, dtype=tf.float32),
        "t": 1,
    }
    result = calc.ell_7(**vars)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell7_calculation() ==================")


def test_ell8_calculation():
    print("\n================== RUNNING: test_ell8_calculation() ==================")
    calc = get_loss_class()
    vars = {
        "lambda_E_t": tf.constant(1.0, dtype=tf.float32),
        "E_t": tf.constant(1.0, dtype=tf.float32),
    }
    result = calc.ell_8(**vars)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell8_calculation() ==================")


def test_ell9_calculation():
    print("\n================== RUNNING: test_ell9_calculation() ==================")
    calc = get_loss_class()
    vars = {
        "lambda_t_BAU": tf.constant(1.0, dtype=tf.float32),
        "E_t": tf.constant(1.0, dtype=tf.float32),
        "k_t": tf.constant(1.0, dtype=tf.float32),
        "t": 1,
    }
    result = calc.ell_9(**vars)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell9_calculation() ==================")


def test_ell10_calculation():
    print("\n================== RUNNING: test_ell10_calculation() ==================")
    calc = get_loss_class()
    s_t = tf.constant(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1], dtype=tf.float32
    )  # Example state at time t
    a_t = tf.constant(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=tf.float32
    )  # Example action at time t
    s_tplus = tf.constant(
        [1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 2], dtype=tf.float32
    )  # Example state at time t+1
    a_tplus = tf.constant(
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=tf.float32
    )  # Example action at time t+1

    # Call the squared_error_for_transition function
    result = calc.squared_error_for_transition(s_t, a_t, s_tplus, a_tplus)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell10_calculation() ==================")


##################### MAIN LOOP #####################


def main():
    print("################## IN MAIN FUNCTION ##################")
    print("\n\n#######################################################")
    print("##################     UNIT TESTS    ##################")
    print("#######################################################\n\n")
    test_ell1_calculation()
    test_ell2_4_calculation()
    test_ell5_6_calculation()
    test_ell7_calculation()
    test_ell8_calculation()
    test_ell9_calculation()
    test_ell10_calculation()

    print("\n\n#######################################################")
    print("################## INTEGRATION TESTS ##################")
    print("#######################################################\n\n")

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
