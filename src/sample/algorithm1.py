"""
Algorithm 1: Basic Gibbs sampling from θ_i | θ_{-i}, y_i
Implementation of Neal (1998) Algorithm 1
"""

import numpy as np
import time
from scipy import stats
from typing import List, Tuple, Optional
from .base import DirichletProcessMixture, MCMCResults
from .base_algorithm import BaseAlgorithm


class Algorithm1(BaseAlgorithm):
    """
    Algorithm 1: Basic Gibbs sampling from θ_i | θ_{-i}, y_i
    State consists of θ_1,...,θ_n only
    
    This algorithm requires conjugacy (F and G0 must be conjugate)
    """

    def initialize_theta(self) -> np.ndarray:
        """Initialize θ by sampling from G0, returns shape (n, d)"""
        return np.array([self.model.sample_prior() for _ in range(self.n)])

    def extract_clusters(self, theta: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract cluster assignments from theta values (vectors)

        Args:
            theta: Array of theta values, shape (n, d)
            tol: Tolerance for considering two vectors equal (Euclidean distance)

        Returns:
            c: Cluster assignments (0-indexed)
            phi: Unique cluster parameters, shape (k, d) where k is number of clusters
        """
        # Use hierarchical clustering based on Euclidean distance
        # For efficiency, we'll use a simple greedy approach
        c = -np.ones(self.n, dtype=int)
        phi_list = []

        for i in range(self.n):
            # Check if theta[i] is close to any existing cluster center
            assigned = False
            for cluster_id, phi_k in enumerate(phi_list):
                dist = np.linalg.norm(theta[i] - phi_k)
                if dist < tol:
                    c[i] = cluster_id
                    assigned = True
                    break

            if not assigned:
                # Create new cluster
                c[i] = len(phi_list)
                phi_list.append(theta[i].copy())

        phi = np.array(phi_list)  # Shape (k, d)
        return c, phi

    def run(self, n_iter: int = 1000, burn_in: int = 100, 
            tol: float = 1e-6, seed: Optional[int] = None) -> MCMCResults:
        """
        Execute Algorithm 1
        
        Args:
            n_iter: Number of iterations after burn-in
            burn_in: Number of burn-in iterations
            tol: Tolerance for cluster identification
            seed: Random seed for reproducibility
            
        Returns:
            MCMCResults containing samples and statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize
        theta = self.initialize_theta()
        
        # Storage
        theta_samples = []
        c_samples = []
        k_samples = []
        
        start_time = time.time()
        
        for iteration in range(n_iter + burn_in):
            # Update each θ_i in random order (improves mixing)
            indices = np.random.permutation(self.n)
            
            for i in indices:
                # Get θ_{-i} (all except i), shape ((n-1), d)
                theta_minus_i = np.delete(theta, i, axis=0)

                # Calculate LOG weights for existing components
                # Note: Use ALL θ_j (j ≠ i), not just unique values
                log_weights = []
                values = []

                for theta_j in theta_minus_i:
                    # log(w_j) = log F(y_i | θ_j)
                    log_w = self.model.log_likelihood(self.y[i], theta_j)
                    log_weights.append(log_w)
                    values.append(theta_j)  # Keep the actual vector

                # Log weight for new component: log(w_new) = log(α) + log(m(y_i))
                log_m_yi = np.sum(stats.norm.logpdf(
                    self.y[i],
                    loc=self.model.mu0,
                    scale=np.sqrt(self.model.sigma2 + self.model.sigma02)
                ))
                log_w_new = np.log(self.alpha) + log_m_yi
                log_weights.append(log_w_new)
                values.append(None)  # Marker for new component

                # Convert to numpy array
                log_weights = np.array(log_weights, dtype=float)

                # Normalize in log-space (subtract max for numerical stability)
                max_log_weight = np.max(log_weights)
                log_weights_normalized = log_weights - max_log_weight
                weights = np.exp(log_weights_normalized)

                # Handle numerical issues
                if np.isnan(weights).any() or weights.sum() == 0:
                    weights = np.ones_like(weights) / len(weights)
                else:
                    weights = weights / weights.sum()

                # Sample new θ_i
                chosen_idx = np.random.choice(len(weights), p=weights)

                if values[chosen_idx] is None:
                    # Sample from H_i (posterior based on y_i alone)
                    theta[i] = self.model.sample_from_H(self.y[i])
                else:
                    # Reuse existing θ_j (vector)
                    theta[i] = values[chosen_idx].copy()
            
            # Store after burn-in
            if iteration >= burn_in:
                theta_samples.append(theta.copy())
                
                # Extract cluster assignments
                c, phi = self.extract_clusters(theta, tol)
                c_samples.append(c.copy())
                k_samples.append(len(phi))
        
        end_time = time.time()
        
        # Convert to arrays
        theta_array = np.array(theta_samples)
        c_array = np.array(c_samples)
        
        # Compute statistics
        k_array = np.array(k_samples)
        
        autocorr_k = self.compute_autocorr(k_array)
        autocorr_theta1 = self.compute_autocorr(theta_array[:, 0])
        
        time_per_iter = (end_time - start_time) / n_iter * 1000  # ms
        
        # Get phi from last iteration
        _, phi_last = self.extract_clusters(theta, tol)
        
        return MCMCResults(
            c=c_array,
            phi=phi_last,
            theta=theta_array,
            time_per_iteration=time_per_iter,
            autocorr_k=autocorr_k,
            autocorr_theta1=autocorr_theta1
        )