"""
Algorithm 5: Metropolis-Hastings with R proposals
"""

import numpy as np
import time
from .base import MCMCResults
from .base_algorithm import BaseAlgorithm


class Algorithm5(BaseAlgorithm):
    """
    Algorithm 5: Metropolis-Hastings updates with R proposals

    For each observation, repeat the following R times:
    1. Draw a candidate c_i* from the conditional prior
    2. Accept/reject using Metropolis-Hastings
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100, R: int = 4) -> MCMCResults:
        """Run Algorithm 5"""
        c, phi, theta = self.initialize()

        # Relabel
        unique_c = np.unique(c)
        c_mapping = {uc: i for i, uc in enumerate(unique_c)}
        c = np.array([c_mapping[ci] for ci in c])
        phi = [phi[uc] for uc in unique_c]

        c_samples = []
        theta_samples = []

        start_time = time.time()

        for iteration in range(n_iter + burn_in):
            # Update each c_i
            for i in range(self.n):
                for r in range(R):
                    # Current assignment
                    current_c = c[i]
                    current_phi = phi[current_c]

                    # Count components excluding i
                    c_temp = np.delete(c, i)
                    n_minus_i_c = np.bincount(c_temp, minlength=len(phi)+1)

                    # Propose new c_i* from prior
                    probs_prior = []
                    for j in range(len(phi)):
                        if n_minus_i_c[j] > 0:
                            probs_prior.append(n_minus_i_c[j])
                        else:
                            probs_prior.append(0)
                    probs_prior.append(self.alpha)

                    probs_prior = np.array(probs_prior)
                    probs_prior /= probs_prior.sum()

                    proposed_c = np.random.choice(len(probs_prior), p=probs_prior)

                    # Create new component if needed
                    if proposed_c == len(phi):
                        proposed_phi = self.model.sample_prior()
                        is_new = True
                    else:
                        proposed_phi = phi[proposed_c]
                        is_new = False

                    # Metropolis-Hastings acceptance
                    log_accept = (self.model.log_likelihood(self.y[i], proposed_phi) -
                                  self.model.log_likelihood(self.y[i], current_phi))

                    if np.log(np.random.random()) < log_accept:
                        # Accept
                        if is_new:
                            phi.append(proposed_phi)
                        c[i] = proposed_c

            # Update phi for non-empty components
            for j in range(len(phi)):
                mask = (c == j)
                if mask.sum() > 0:
                    y_c = self.y[mask]  # Shape (n_c, d)
                    n_c = len(y_c)
                    # For diagonal covariance, update each dimension independently
                    prec_post = n_c / self.model.sigma2 + 1 / self.model.sigma02
                    mu_post = (y_c.sum(axis=0) / self.model.sigma2 + self.model.mu0 / self.model.sigma02) / prec_post
                    sigma_post = 1 / np.sqrt(prec_post)
                    phi[j] = np.random.normal(mu_post, sigma_post)

            # Remove empty components - only do this periodically to avoid constant relabeling
            # This improves efficiency and reduces potential inconsistencies
            if (iteration + 1) % 10 == 0 or iteration == n_iter + burn_in - 1:
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

        autocorr_k = self.compute_autocorr(k_samples)
        autocorr_theta1 = self.compute_autocorr(theta_array[:, 0])
        time_per_iter = (end_time - start_time) / n_iter * 1000

        return MCMCResults(
            c=c_array,
            phi=np.array(phi),
            theta=theta_array,
            time_per_iteration=time_per_iter,
            autocorr_k=autocorr_k,
            autocorr_theta1=autocorr_theta1
        )
