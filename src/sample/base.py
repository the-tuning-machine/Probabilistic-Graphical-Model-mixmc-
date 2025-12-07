"""
Base classes for Dirichlet Process Mixture Models
Based on: Neal, R. M. (1998). "Markov Chain Sampling Methods for Dirichlet Process Mixture Models"

Author: Implementation based on the paper
Date: 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple
from dataclasses import dataclass


@dataclass
class MCMCResults:
    """Store results from MCMC sampling"""
    c: np.ndarray  # Component indicators
    phi: np.ndarray  # Component parameters
    theta: np.ndarray  # Individual parameters
    time_per_iteration: float
    autocorr_k: float
    autocorr_theta1: float


class DirichletProcessMixture:
    """
    Dirichlet Process Mixture Model

    Model:
    y_i | θ_i ~ F(θ_i)
    θ_i | G ~ G
    G ~ DP(G_0, α)

    For the example in the paper:
    F(θ) = N(θ, σ²) with σ² = 0.01
    G_0 = N(0, 1)
    α = 1
    """

    def __init__(self, data: np.ndarray, alpha: float = 1.0,
                 sigma: float = 0.1, mu0: float = 0.0, sigma0: float = 1.0):
        """
        Initialize the Dirichlet Process Mixture Model

        Args:
            data: Observed data (y_i)
            alpha: Concentration parameter
            sigma: Standard deviation of F(θ) = N(θ, σ²)
            mu0: Mean of G_0 = N(μ_0, σ_0²)
            sigma0: Standard deviation of G_0
        """
        self.y = np.array(data)
        self.n = len(data)
        self.alpha = alpha
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma02 = sigma0 ** 2

    def likelihood(self, y: float, theta: float) -> float:
        """F(y_i | θ_i) = N(y_i; θ_i, σ²)"""
        return stats.norm.pdf(y, loc=theta, scale=self.sigma)

    def log_likelihood(self, y: float, theta: float) -> float:
        """log F(y_i | θ_i)"""
        return stats.norm.logpdf(y, loc=theta, scale=self.sigma)

    def prior(self, theta: float) -> float:
        """G_0(θ) = N(θ; μ_0, σ_0²)"""
        return stats.norm.pdf(theta, loc=self.mu0, scale=self.sigma0)

    def log_prior(self, theta: float) -> float:
        """log G_0(θ)"""
        return stats.norm.logpdf(theta, loc=self.mu0, scale=self.sigma0)

    def sample_prior(self) -> float:
        """Sample θ ~ G_0"""
        return np.random.normal(self.mu0, self.sigma0)

    def posterior_theta(self, y: float) -> Tuple[float, float]:
        """
        Compute posterior parameters for θ | y under conjugate prior

        Posterior: N(θ; μ_post, σ_post²)
        where:
        1/σ_post² = 1/σ² + 1/σ_0²
        μ_post/σ_post² = y/σ² + μ_0/σ_0²

        Returns:
            (μ_post, σ_post)
        """
        prec_post = 1/self.sigma2 + 1/self.sigma02
        sigma_post = 1 / np.sqrt(prec_post)
        mu_post = (y/self.sigma2 + self.mu0/self.sigma02) / prec_post
        return mu_post, sigma_post

    def sample_posterior_theta(self, y: float) -> float:
        """Sample θ | y from posterior"""
        mu_post, sigma_post = self.posterior_theta(y)
        return np.random.normal(mu_post, sigma_post)
