"""
Base class for MCMC algorithms
"""

import numpy as np
from typing import List, Tuple
from .base import DirichletProcessMixture


class BaseAlgorithm:
    """Base class for MCMC algorithms"""

    def __init__(self, model: DirichletProcessMixture):
        self.model = model
        self.y = model.y
        self.n = model.n
        self.alpha = model.alpha

    def initialize(self) -> Tuple[np.ndarray, List[float], np.ndarray]:
        """
        Initialize component indicators and parameters

        Returns:
            c: Component indicators (shape: n)
            phi: List of unique component parameters
            theta: Individual parameters (shape: n)
        """
        # Start with each observation in its own component
        c = np.arange(self.n)
        phi = [self.model.sample_posterior_theta(yi) for yi in self.y]
        theta = np.array(phi)
        return c, phi, theta

    def compute_autocorr(self, samples: np.ndarray, lag: int = 1) -> float:
        """Compute autocorrelation at given lag"""
        if len(samples) < lag + 1:
            return 0.0
        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)
        if var == 0:
            return 0.0
        autocov = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean))
        return autocov / var
