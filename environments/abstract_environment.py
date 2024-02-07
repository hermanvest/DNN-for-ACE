from abc import ABC, abstractmethod
from typing import Any
import tensorflow as tf
import numpy as np


class Abstract_Environment(ABC):
    @abstractmethod
    def step(self, policy_output: tf.Tensor) -> Any:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def compute_loss(self, batch: np.ndarray) -> Any:
        pass
