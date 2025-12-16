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

    def initialize(self) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Initialize component indicators and parameters

        Returns:
            c: Component indicators (shape: n)
            phi: List of unique component parameters (each is a vector of shape (d,))
            theta: Individual parameters (shape: (n, d))
        """
        # Start with each observation in its own component
        c = np.arange(self.n)
        phi = [self.model.sample_posterior_theta(self.y[i]) for i in range(self.n)]
        theta = np.array(phi)  # Shape (n, d)
        return c, phi, theta

    def compute_autocorr(self, samples: np.ndarray) -> float:
        """
        Compute autocorrelation time (integrated autocorrelation time).

        This is what Neal (1998) reports in Table 1, not simple autocorrelation.
        The autocorrelation time τ measures how many iterations are needed
        to get effectively independent samples.

        τ = 1 + 2 * Σ_{k=1}^{K} ρ_k

        where we sum until autocorrelations become negligible.
        """
        if len(samples) < 10:
            return 1.0

        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)

        if var == 0:
            return 1.0

        # Compute autocorrelations up to a reasonable lag
        max_lag = min(n // 2, 500)  # Don't go beyond n/2 or 500
        autocorr_time = 1.0  # τ starts at 1

        for k in range(1, max_lag):
            if k >= n:
                break

            # Compute autocorrelation at lag k
            autocov = np.mean((samples[:n-k] - mean) * (samples[k:] - mean))
            rho_k = autocov / var

            # Add 2*ρ_k to the sum
            autocorr_time += 2 * rho_k

            # Stop if autocorrelation becomes very small
            # (Sokal's criterion: stop when k > 5*τ_current)
            if k > 5 * autocorr_time:
                break

            # Also stop if autocorrelation becomes negative
            if rho_k < 0:
                break

        return max(1.0, autocorr_time)  # τ should be at least 1
