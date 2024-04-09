import numpy as np
import tensorflow as tf
from pathlib import Path

from environments.ace_dice.equations_of_motion.eom_ace_dice_2016 import (
    Eom_Ace_Dice_2016,
)
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


def get_equations() -> Eom_Ace_Dice_2016:
    configs = load_config(yaml_file_path)
    states = configs["state_variables"]
    actions = configs["action_variables"]
    parameters = configs["parameters"]
    return Eom_Ace_Dice_2016(t_max, states, actions, parameters)


def test_constant_creations():
    print("============ Running test_constant_creations ============")
    eq = get_equations()
    print(f"The list of labor is: {eq.N_t.numpy()}")
    print(f"The list of tfp is: {eq.A_t.numpy()}")
    print(f"The list of carbon intensity is: {eq.sigma.numpy()}")
    print(f"The list of abatement costs is: {eq.theta_1.numpy()}")
    print("============ End of test_constant_creations ============\n\n")


def test_y_gross():
    print("============ Running test_y_gross ============")
    eq = get_equations()

    print("Creating constants...")
    k_t = tf.constant(1.0, dtype=tf.float32)
    t = 1

    print("Calculating y_gross...")
    y_t_gross = eq.Y_gross(t, k_t)

    print(f"The value of y_gross is: {y_t_gross}")
    print("============ End of test_y_gross ============\n\n")


def test_E_t_BAU():
    print("============ Running test_E_t_BAU ============")
    eq = get_equations()

    print("Creating constants...")
    k_t = tf.constant(1.0, dtype=tf.float32)
    t = 1

    print("Calculating e_t_BAU...")
    e_t_BAU = eq.E_t_BAU(t, k_t)

    print(f"The value of E_t_BAU is: {e_t_BAU}")
    print("============ End of test_E_t_BAU ============\n\n")


def test_log_Y_t():
    print("============ Running test_log_Y_t ============")
    eq = get_equations()

    print("Creating constants...")
    k_t = tf.constant(1.0, dtype=tf.float32)
    E_t = tf.constant(1.0, dtype=tf.float32)
    t = 1

    print("Calculating log_Y_t...")
    log_Y_t = eq.log_Y_t(k_t, E_t, t)

    print(f"The value of log_Y_t is: {log_Y_t}")
    print("============ End of test_log_Y_t ============\n\n")


def test_calculation_of_k_t_plus():
    print("============ Running test_calculation_of_k_t_plus ============")
    eq = get_equations()

    print("Creating constants...")
    k_t = tf.constant(1.0, dtype=tf.float32)
    tau_1_t = tf.constant(1.0, dtype=tf.float32)
    E_t = tf.constant(1.0, dtype=tf.float32)
    x_t = tf.constant(0.6, dtype=tf.float32)
    t = 1

    print("Calculating log_Y_t...")
    log_Y_t = eq.log_Y_t(k_t, E_t, t)

    print("Calculating k_tplus...")
    k_tplus = eq.k_tplus(log_Y_t, tau_1_t, x_t, t)
    print(f"The value of capital for the next period is: {k_tplus}")
    print("============ End of test_calculation_of_k_t_plus ============\n\n")


def test_calculation_of_m_1_plus():
    print("============ Running test_calculation_of_m_1_plus ============")
    eq = get_equations()

    print("Creating constants...")
    E_t = tf.constant(1.0, dtype=tf.float32)
    m = tf.ones([3], dtype=tf.float32)
    k_t = tf.constant(1.0, dtype=tf.float32)
    t = 1

    print("Calculating m_1plus...")
    print(
        f" The value of the carbon stock in the next period is {eq.m_1plus(m, E_t, k_t, t)}"
    )
    print("============ End of test_calculation_of_m_1_plus ============\n\n")


def test_calculation_of_m_2_plus():
    print("============ Running test_calculation_of_m_2_plus ============")
    eq = get_equations()

    print("Creating constants...")
    m = tf.ones([3], dtype=tf.float32)

    print("Calculating m_2plus...")
    print(f" The value of the carbon stock in the next period is {eq.m_2plus(m)}")
    print("============ End of test_calculation_of_m_2_plus ============\n\n")


def test_calculation_of_m_3_plus():
    print("============ Running test_calculation_of_m_3_plus ============")
    eq = get_equations()

    print("Creating constants...")
    m = tf.ones([3], dtype=tf.float32)

    print("Calculating m_3plus...")
    print(f" The value of the carbon stock in the next period is {eq.m_3plus(m)}")
    print("============ End of test_calculation_of_m_3_plus ============\n\n")


def test_calculation_of_tau_1plus():
    print("============ Running test_calculation_of_tau_1plus ============")
    eq = get_equations()
    m_1_t = tf.constant(1.0, dtype=tf.float32)
    tau_t = tf.ones([2], dtype=tf.float32)
    print(
        f" The value of the transformed temperatures in the next period is {eq.tau_1plus(tau_t, m_1_t)}"
    )
    print("============ End of test_calculation_of_tau_1plus ============\n\n")


def test_calculation_of_tau_2plus():
    print("============ Running test_calculation_of_tau_2plus ============")
    eq = get_equations()
    tau_t = tf.ones([2], dtype=tf.float32)
    print(
        f" The value of the transformed temperatures in the next period is {eq.tau_2plus(tau_t)}"
    )
    print("============ End of test_calculation_of_tau_2plus ============\n\n")


def test_update_state():
    print("============ Running test_update_state ============")
    eq = get_equations()

    print("Creating a state sample...")
    s_t = [1, 1, 1, 1, 1, 1, 1]
    s_t = tf.convert_to_tensor(s_t, dtype=tf.float32)

    print("Creating an action sample...")
    a_t = [0.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 11 actions
    a_t = tf.convert_to_tensor(a_t, dtype=tf.float32)

    print("Generating the next state and printing values...")
    s_tplus = eq.update_state(s_t, a_t)
    print(f"Current action variables: {a_t}")
    print(f"Current state variables: {s_t}")
    print(f"Next state variables: {s_tplus}")

    print("============ End of test_update_state ============\n\n")


def main():
    print("##################    CONSTANTS     ##################")
    test_constant_creations()

    print("################## HELPER FUNCTIONS ##################")
    test_y_gross()
    test_E_t_BAU()
    test_log_Y_t()

    print("##################   CALCULATIONS   ##################")
    test_calculation_of_k_t_plus()
    test_calculation_of_m_1_plus()
    test_calculation_of_m_2_plus()
    test_calculation_of_m_3_plus()
    test_calculation_of_tau_1plus()
    test_calculation_of_tau_2plus()
    test_update_state()


if __name__ == "__main__":
    main()
