import itertools
import numpy as np
from .state_feature_mapping import StateFeatureMapping


class FourierStateFeatureMapping(StateFeatureMapping):
    """Approximates observations using Fourier basis functions."""
    def __init__(self, low: np.ndarray, high: np.ndarray, order: int):
        super().__init__()

        # Boundary of environment observation space.
        self.n_dimensions = low.size
        self.low = low
        self.high = high

        # Number of terms in Fourier series.
        self.order = order

        # Compute coefficient matrix.
        # These are vectors of value `c` which are 
        # systematically varied between [0, n_dimensions].
        iter = itertools.product(range(order+1), repeat=self.n_dimensions)
        self.coeffs = np.array([list(map(int,x)) for x in iter])
        self.n_features = self.coeffs.shape[0]

    def __call__(self, state: np.ndarray):
        """Compute approximation for given observation."""

        # Scale the observation.
        scaled = (state - self.low)/(self.high - self.low)

        # Compute approximation using Fourier basis functions.
        return np.cos(np.pi * np.dot(self.coeffs, scaled))