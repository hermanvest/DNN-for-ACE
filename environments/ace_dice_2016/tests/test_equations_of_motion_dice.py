# For running, do: pytest -p no:warnings
import pytest
import numpy as np
from pathlib import Path

from environments.ace_dice_2016.equations_of_motion import Equations_of_motion_Ace_Dice
from utils.config_loader import load_config

current_script_path = Path(__file__).parent

# Navigate up to the common root and then to the YAML file
yaml_file_path = (
    current_script_path.parent.parent.parent
    / "configs"
    / "state_and_action_space"
    / "ace_dice_2016.yaml"
)

t_max = 10


@pytest.fixture
def setUp():
    configs = load_config(yaml_file_path)
    states = configs["state_variables"]
    actions = configs["action_variables"]
    parameters = configs["parameters"]
    return Equations_of_motion_Ace_Dice(t_max, states, actions, parameters)


def test_carbon_intensity_creation_length(setUp):
    assert t_max == setUp.sigma.size


def test_abatement_cost_creation_length(setUp):
    assert t_max == setUp.theta_1.size


def test_labor_creation_length(setUp):
    assert t_max == setUp.N_t.size


def test_tfp_creation_length(setUp):
    assert t_max == setUp.a_t.size
