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


def test_carbon_intensity_creation():
    eq = get_equations()
    print("============ Running test_carbon_intensity_creation ============")
    print(f" Size of list: {eq.sigma.size}")
    for sigma_t in eq.sigma:
        print(sigma_t)

    print("============ End of test_carbon_intensity_creation ============\n\n")


def test_labor_creation():
    print("============ Running test_labor_creation ============")
    eq = get_equations()

    print("============ End of test_labor_creation ============\n\n")


def main():
    test_carbon_intensity_creation()
    test_labor_creation()


if __name__ == "__main__":
    main()
