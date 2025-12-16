"""
Base classes for Dirichlet Process Mixture Models
Based on: Neal, R. M. (1998). "Markov Chain Sampling Methods for Dirichlet Process Mixture Models"

Supports multivariate Gaussian data with diagonal covariance matrices.

Author: Implementation based on the paper
Date: 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple, Union
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
    Dirichlet Process Mixture Model (Multivariate Gaussian)

    Model:
    y_i | θ_i ~ F(θ_i) = N(θ_i, Σ)
    θ_i | G ~ G
    G ~ DP(G_0, α)
    G_0 = N(μ_0, Σ_0)

    We use diagonal covariance matrices for simplicity:
    - Σ = diag(σ²) where σ² is a scalar or vector
    - Σ_0 = diag(σ_0²) where σ_0² is a scalar or vector
    """

    def __init__(self, data: np.ndarray, alpha: float = 1.0,
                 sigma: Union[float, np.ndarray] = 0.1,
                 mu0: Union[float, np.ndarray] = 0.0,
                 sigma0: Union[float, np.ndarray] = 1.0):
        """
        Initialize the Dirichlet Process Mixture Model

        Args:
            data: Observed data, shape (n, d) for multivariate or (n,) for univariate
            alpha: Concentration parameter
            sigma: Standard deviation of F(θ) = N(θ, Σ).
                   Scalar or array of length d for diagonal covariance
            mu0: Mean of G_0 = N(μ_0, Σ_0). Scalar or array of length d
            sigma0: Standard deviation of G_0. Scalar or array of length d
        """
        self.y = np.atleast_2d(data)
        if self.y.shape[0] == 1 and len(data.shape) == 1:
            # If data was 1D, transpose to (n, 1)
            self.y = data.reshape(-1, 1)

        self.n, self.d = self.y.shape
        self.alpha = alpha

        # Convert sigma to array
        if np.isscalar(sigma):
            self.sigma = np.full(self.d, sigma)
        else:
            self.sigma = np.array(sigma)
            assert len(self.sigma) == self.d, f"sigma must have length {self.d}"

        self.sigma2 = self.sigma ** 2

        # Convert mu0 to array
        if np.isscalar(mu0):
            self.mu0 = np.full(self.d, mu0)
        else:
            self.mu0 = np.array(mu0)
            assert len(self.mu0) == self.d, f"mu0 must have length {self.d}"

        # Convert sigma0 to array
        if np.isscalar(sigma0):
            self.sigma0 = np.full(self.d, sigma0)
        else:
            self.sigma0 = np.array(sigma0)
            assert len(self.sigma0) == self.d, f"sigma0 must have length {self.d}"

        self.sigma02 = self.sigma0 ** 2

    def likelihood(self, y: np.ndarray, theta: np.ndarray) -> float:
        """
        F(y_i | θ_i) = N(y_i; θ_i, Σ)

        For diagonal covariance, this is the product of univariate normals.
        """
        # Ensure vectors
        y = np.atleast_1d(y)
        theta = np.atleast_1d(theta)

        # Product of independent normals (diagonal covariance)
        log_prob = np.sum(stats.norm.logpdf(y, loc=theta, scale=self.sigma))
        return np.exp(log_prob)

    def log_likelihood(self, y: np.ndarray, theta: np.ndarray) -> float:
        """log F(y_i | θ_i)"""
        y = np.atleast_1d(y)
        theta = np.atleast_1d(theta)
        return np.sum(stats.norm.logpdf(y, loc=theta, scale=self.sigma))

    def prior(self, theta: np.ndarray) -> float:
        """G_0(θ) = N(θ; μ_0, Σ_0)"""
        theta = np.atleast_1d(theta)
        log_prob = np.sum(stats.norm.logpdf(theta, loc=self.mu0, scale=self.sigma0))
        return np.exp(log_prob)

    def log_prior(self, theta: np.ndarray) -> float:
        """log G_0(θ)"""
        theta = np.atleast_1d(theta)
        return np.sum(stats.norm.logpdf(theta, loc=self.mu0, scale=self.sigma0))

    def sample_prior(self) -> np.ndarray:
        """Sample θ ~ G_0, returns vector of shape (d,)"""
        return np.random.normal(self.mu0, self.sigma0)

    def posterior_theta(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior parameters for θ | y under conjugate prior

        For diagonal covariances, each dimension is independent:
        Posterior: N(θ; μ_post, Σ_post)
        where for each dimension j:
        1/σ²_post[j] = 1/σ²[j] + 1/σ²_0[j]
        μ_post[j]/σ²_post[j] = y[j]/σ²[j] + μ_0[j]/σ²_0[j]

        Returns:
            (μ_post, σ_post): Both are vectors of shape (d,)
        """
        y = np.atleast_1d(y)

        prec_post = 1/self.sigma2 + 1/self.sigma02
        sigma_post = 1 / np.sqrt(prec_post)
        mu_post = (y/self.sigma2 + self.mu0/self.sigma02) / prec_post
        return mu_post, sigma_post

    def sample_posterior_theta(self, y: np.ndarray) -> np.ndarray:
        """Sample θ | y from posterior, returns vector of shape (d,)"""
        mu_post, sigma_post = self.posterior_theta(y)
        return np.random.normal(mu_post, sigma_post)

    def marginal_likelihood(self, y: np.ndarray) -> float:
        """
        Marginal likelihood: m(y) = ∫ F(y|θ) G0(dθ)

        For diagonal normal-normal model:
        m(y) = ∏_j N(y[j] | μ0[j], σ²[j] + σ²_0[j])
        """
        y = np.atleast_1d(y)
        variance = self.sigma2 + self.sigma02
        log_prob = np.sum(stats.norm.logpdf(y, loc=self.mu0, scale=np.sqrt(variance)))
        return np.exp(log_prob)

    def sample_from_H(self, y: np.ndarray) -> np.ndarray:
        """
        Sample from H_i(θ) ∝ F(y_i|θ) G0(dθ)

        For conjugate normal-normal model, this is the posterior:
        H_i = N(θ | μ_post, Σ_post)
        Same as sample_posterior_theta, but kept for clarity
        """
        return self.sample_posterior_theta(y)