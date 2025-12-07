"""
Algorithm 3: "No Gaps" Algorithm
"""

import numpy as np
from scipy import stats
import time
from .base import MCMCResults
from .base_algorithm import BaseAlgorithm


class Algorithm3(BaseAlgorithm):
    """
    Algorithm 3: "No gaps" algorithm

    Maintains components numbered 1,...,k without gaps.
    Similar to Algorithm 1 but with careful relabeling.
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100) -> MCMCResults:
        """Run Algorithm 3"""
        c, phi, theta = self.initialize()

        # Relabel to remove gaps
        unique_c = np.unique(c)
        c_mapping = {uc: i for i, uc in enumerate(unique_c)}
        c = np.array([c_mapping[ci] for ci in c])
        phi = [phi[uc] for uc in unique_c]

        c_samples = []
        theta_samples = []

        start_time = time.time()

        for iteration in range(n_iter + burn_in):
            k = len(phi)

            # Update each c_i
            for i in range(self.n):
                # Count components excluding i
                c_temp = np.delete(c, i)
                n_minus_i_c = np.bincount(c_temp, minlength=k+1)

                # Probabilities for existing components
                probs = []
                for j in range(k):
                    if n_minus_i_c[j] > 0:
                        prob = n_minus_i_c[j] * self.model.likelihood(self.y[i], phi[j])
                    else:
                        prob = 0
                    probs.append(prob)

                # Probability for new component
                sigma_marg = np.sqrt(self.model.sigma2 + self.model.sigma02)
                prob_new = self.alpha * stats.norm.pdf(self.y[i], self.model.mu0, sigma_marg)
                probs.append(prob_new)

                # Sample
                probs = np.array(probs)
                probs /= probs.sum()
                new_c = np.random.choice(len(probs), p=probs)

                if new_c == k:
                    # New component
                    phi.append(self.model.sample_posterior_theta(self.y[i]))
                    c[i] = k
                else:
                    c[i] = new_c

            # Update phi
            k = len(phi)
            for j in range(k):
                mask = (c == j)
                if mask.sum() > 0:
                    y_c = self.y[mask]
                    n_c = len(y_c)
                    prec_post = n_c / self.model.sigma2 + 1 / self.model.sigma02
                    mu_post = (y_c.sum() / self.model.sigma2 + self.model.mu0 / self.model.sigma02) / prec_post
                    sigma_post = 1 / np.sqrt(prec_post)
                    phi[j] = np.random.normal(mu_post, sigma_post)

            # Remove empty components
            unique_c = np.unique(c)
            if len(unique_c) < len(phi):
                new_phi = [phi[j] for j in unique_c]
                c_mapping = {uc: i for i, uc in enumerate(unique_c)}
                c = np.array([c_mapping[ci] for ci in c])
                phi = new_phi

            theta = np.array([phi[ci] for ci in c])

            if iteration >= burn_in:
                c_samples.append(c.copy())
                theta_samples.append(theta.copy())

        end_time = time.time()

        c_array = np.array(c_samples)
        theta_array = np.array(theta_samples)
        k_samples = np.array([len(np.unique(c_sample)) for c_sample in c_samples])

        autocorr_k = self.compute_autocorr(k_samples, lag=1)
        autocorr_theta1 = self.compute_autocorr(theta_array[:, 0], lag=1)
        time_per_iter = (end_time - start_time) / n_iter * 1000

        return MCMCResults(
            c=c_array,
            phi=np.array(phi),
            theta=theta_array,
            time_per_iteration=time_per_iter,
            autocorr_k=autocorr_k,
            autocorr_theta1=autocorr_theta1
        )
