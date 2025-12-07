"""
Algorithm 1: Basic Gibbs Sampling for Dirichlet Process Mixture
"""

import numpy as np
from scipy import stats
import time
from .base import MCMCResults
from .base_algorithm import BaseAlgorithm


class Algorithm1(BaseAlgorithm):
    """
    Algorithm 1: Basic Gibbs sampling for Dirichlet Process Mixture

    At each iteration:
    1. For i = 1,...,n: Update c_i by sampling from full conditional
    2. For each unique c: Update φ_c by sampling from posterior
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100) -> MCMCResults:
        """Run Algorithm 1"""
        c, phi, theta = self.initialize()

        # Storage for samples
        c_samples = []
        theta_samples = []

        start_time = time.time()

        for iteration in range(n_iter + burn_in):
            # Update each c_i
            for i in range(self.n):
                # Remove observation i from its current component
                old_c = c[i]
                c_temp = np.delete(c, i)

                # Count components (excluding i)
                unique_c, counts = np.unique(c_temp, return_counts=True)
                n_minus_i = {uc: count for uc, count in zip(unique_c, counts)}

                # Compute probabilities for existing components
                probs = []
                components = []

                for uc in unique_c:
                    if uc in n_minus_i:
                        count = n_minus_i[uc]
                        # Probability proportional to: count * F(y_i | φ_c)
                        prob = count * self.model.likelihood(self.y[i], phi[uc])
                        probs.append(prob)
                        components.append(uc)

                # Probability for new component
                # Sample φ_new ~ G_0 and compute α * ∫ F(y_i | φ) dG_0(φ)
                # For conjugate case: α * F_marginal(y_i)
                mu_post, sigma_post = self.model.posterior_theta(self.y[i])
                # Marginal likelihood: N(y_i; μ_0, σ² + σ_0²)
                sigma_marg = np.sqrt(self.model.sigma2 + self.model.sigma02)
                prob_new = self.alpha * stats.norm.pdf(self.y[i], self.model.mu0, sigma_marg)
                probs.append(prob_new)
                components.append(len(phi))  # New component index

                # Normalize probabilities
                probs = np.array(probs)
                probs /= probs.sum()

                # Sample new component
                new_c = np.random.choice(components, p=probs)
                c[i] = new_c

                # If new component, create new phi
                if new_c == len(phi):
                    phi.append(self.model.sample_posterior_theta(self.y[i]))

            # Update phi for each component
            unique_c = np.unique(c)
            new_phi = []
            c_mapping = {}
            for new_idx, uc in enumerate(unique_c):
                mask = (c == uc)
                y_c = self.y[mask]
                # Sample from posterior given all y in component
                # For normal-normal conjugate:
                n_c = len(y_c)
                prec_post = n_c / self.model.sigma2 + 1 / self.model.sigma02
                mu_post = (y_c.sum() / self.model.sigma2 + self.model.mu0 / self.model.sigma02) / prec_post
                sigma_post = 1 / np.sqrt(prec_post)
                new_phi.append(np.random.normal(mu_post, sigma_post))
                c_mapping[uc] = new_idx

            # Relabel components
            c = np.array([c_mapping[ci] for ci in c])
            phi = new_phi

            # Update theta
            theta = np.array([phi[ci] for ci in c])

            # Store samples after burn-in
            if iteration >= burn_in:
                c_samples.append(c.copy())
                theta_samples.append(theta.copy())

        end_time = time.time()

        # Compute statistics
        c_array = np.array(c_samples)
        theta_array = np.array(theta_samples)

        # Number of distinct components
        k_samples = np.array([len(np.unique(c_sample)) for c_sample in c_samples])

        # Autocorrelations
        autocorr_k = self.compute_autocorr(k_samples, lag=1)
        autocorr_theta1 = self.compute_autocorr(theta_array[:, 0], lag=1)

        time_per_iter = (end_time - start_time) / n_iter * 1000  # milliseconds

        return MCMCResults(
            c=c_array,
            phi=np.array(phi),
            theta=theta_array,
            time_per_iteration=time_per_iter,
            autocorr_k=autocorr_k,
            autocorr_theta1=autocorr_theta1
        )
