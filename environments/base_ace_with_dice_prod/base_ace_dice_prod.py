from typing import Any, Dict


class Base_ace_dice_prod:
    metadata = {"render_modes": ["none", "human"]}

    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    def _get_obs(self) -> None:
        pass

    def _update_state(self) -> None:
        pass

    def step(self) -> None:
        """
        Takes as input, the action obtained from the policy.
        Returns the state we trantision to by taking the step in the environment.
        """
        pass

    def render(self) -> None:
        pass

    def reset(self) -> None:
        """
        Resets state to the initial states in the config file.
        Returns initial states.
        """
        pass

    def close(self) -> None:
        """
        Terminates environment
        """
        pass
