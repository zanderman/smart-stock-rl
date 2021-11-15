
from numpy import ndarray


class StateFeatureMapping:
    def __init__(self):
        pass

    def __call__(self, state: ndarray):
        """Compute approximation for given observation."""
        raise NotImplementedError