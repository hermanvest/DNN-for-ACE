import numpy as np
from pathlib import Path

from environments.ace_dice_2016.equations_of_motion import Equations_of_motion_Ace_Dice
from environments.ace_dice_2016.compute_loss import Computeloss

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
def get_equations() -> Equations_of_motion_Ace_Dice:
    configs = load_config(yaml_file_path)
    states = configs["state_variables"]
    actions = configs["action_variables"]
    parameters = configs["parameters"]
    return Equations_of_motion_Ace_Dice(t_max, states, actions, parameters)


def get_loss_class() -> Computeloss:
    configs = load_config(yaml_file_path)
    parameters = configs["parameters"]
    return Computeloss(parameters, get_equations())


#####################  CALCULATION TESTS #####################
def test_ell1_calculation():
    print("\n================== RUNNING: test_ell1_calculation() ==================")

    calc = get_loss_class()
    lambda_k_t = 2
    result = calc.ell_1(lambda_k_t)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell1_calculation() ==================")


def test_ell2_4_calculation():
    print("\n================== RUNNING: test_ell2_4_calculation() ==================")
    calc = get_loss_class()
    lambda_m_t = np.array([1, 1, 1])
    lambda_tau_1_t = 1
    result = calc.ell_2_4(lambda_m_t, lambda_tau_1_t)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell2_4_calculation() ==================")


def test_ell5_6_calculation():
    print("\n================== RUNNING: test_ell5_6_calculation() ==================")
    calc = get_loss_class()
    lambda_tau_t_vector = np.array([1, 1])
    result = calc.ell_5_6(lambda_tau_t_vector)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell5_6_calculation() ==================")


def test_ell7_calculation():
    print("\n================== RUNNING: test_ell7_calculation() ==================")
    calc = get_loss_class()
    vars = {
        "k_tplus": 3.0,
        "x_t": 0.6,
        "E_t": 1.0,
        "k_t": 1.0,
        "tau_1_t": 1.0,
        "t": 1,
    }
    result = calc.ell_7(**vars)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell7_calculation() ==================")


def test_ell8_calculation():
    print("\n================== RUNNING: test_ell8_calculation() ==================")
    calc = get_loss_class()
    vars = {
        "lambda_E_t": 1,
        "E_t": 1.0,
    }
    result = calc.ell_8(**vars)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell8_calculation() ==================")


def test_ell9_calculation():
    print("\n================== RUNNING: test_ell9_calculation() ==================")
    calc = get_loss_class()
    vars = {
        "lambda_t_BAU": 1,
        "E_t": 1.0,
        "k_t": 1.0,
        "t": 1,
    }
    result = calc.ell_9(**vars)
    print(f"Result: {result}")
    print("================== TERMINATES: test_ell9_calculation() ==================")


##################### MAIN LOOP #####################


def main():
    print("================== IN MAIN FUNCTION ==================")
    test_ell1_calculation()
    test_ell2_4_calculation()
    test_ell5_6_calculation()
    test_ell7_calculation()
    test_ell8_calculation()
    test_ell9_calculation()

    print("================== END TESTS ==================")


if __name__ == "__main__":
    main()
