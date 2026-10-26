from abc import ABC, abstractmethod
import numpy as np


class SIM_ENV(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError("step method must be implemented by subclass.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset method must be implemented by subclass.")

    @staticmethod
    def cossin(vec1, vec2):
        """
        Compute the cosine and sine of the angle between two 2D vectors.

        Args:
            vec1 (list): First 2D vector.
            vec2 (list): Second 2D vector.

        Returns:
            (tuple): (cosine, sine) of the angle between the vectors.
        """
        vec1 = vec1 / (np.linalg.norm(vec1) + 0.00000000001)
        vec2 = vec2 / (np.linalg.norm(vec2) + 0.00000000001)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    @staticmethod
    @abstractmethod
    def get_reward():
        raise NotImplementedError("get_reward method must be implemented by subclass.")
