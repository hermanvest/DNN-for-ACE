import numpy as np
from pathlib import Path

from environments.ace_dice_2016.equations_of_motion import Equations_of_motion_Ace_Dice
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


def get_equations() -> Equations_of_motion_Ace_Dice:
    configs = load_config(yaml_file_path)
    states = configs["state_variables"]
    actions = configs["action_variables"]
    parameters = configs["parameters"]
    return Equations_of_motion_Ace_Dice(t_max, states, actions, parameters)


def test_calculation_of_k_t_plus():
    eq = get_equations()
    print("============ Running test_calculation_of_k_t_plus ============")
    k_t = 1
    tau_1_t = 1
    E_t = eq.E_t_BAU(1, 1) - 1e-5
    t = 1

    log_Y_t = eq.log_Y_t(k_t, E_t, t)
    k_tplus = eq.k_tplus(log_Y_t, tau_1_t, 0.5, 1)
    print(f"The value of capital for the next period is: {k_tplus}")
    print("============ End of test_calculation_of_k_t_plus ============\n\n")


def test_calculation_of_m_1_plus():
    print("============ Running test_calculation_of_m_1_plus ============")
    eq = get_equations()
    m = np.ones(3)
    print(f" The value of the carbon stock in the next period is {eq.m_1plus(m, 1)}")
    print("============ End of test_calculation_of_m_1_plus ============\n\n")


def test_calculation_of_m_2_plus():
    print("============ Running test_calculation_of_m_2_plus ============")
    eq = get_equations()
    m = np.ones(3)
    print(f" The value of the carbon stock in the next period is {eq.m_2plus(m)}")
    print("============ End of test_calculation_of_m_2_plus ============\n\n")


def test_calculation_of_m_3_plus():
    print("============ Running test_calculation_of_m_3_plus ============")
    eq = get_equations()
    m = np.ones(3)
    print(f" The value of the carbon stock in the next period is {eq.m_3plus(m)}")
    print("============ End of test_calculation_of_m_3_plus ============\n\n")


def test_calculation_of_tau_1plus():
    print("============ Running test_calculation_of_tau_1plus ============")
    eq = get_equations()
    tau_t = [1, 1]
    print(
        f" The value of the carbon stock in the next period is {eq.tau_1plus(tau_t,1)}"
    )
    print("============ End of test_calculation_of_tau_1plus ============\n\n")


def test_calculation_of_tau_2plus():
    print("============ Running test_calculation_of_tau_2plus ============")
    eq = get_equations()
    tau_t = [1, 1]
    print(f" The value of the carbon stock in the next period is {eq.tau_2plus(tau_t)}")
    print("============ End of test_calculation_of_tau_2plus ============\n\n")


def main():
    test_calculation_of_k_t_plus()
    test_calculation_of_m_1_plus()
    test_calculation_of_m_2_plus()
    test_calculation_of_m_3_plus()
    test_calculation_of_tau_1plus()
    test_calculation_of_tau_2plus()


if __name__ == "__main__":
    main()
