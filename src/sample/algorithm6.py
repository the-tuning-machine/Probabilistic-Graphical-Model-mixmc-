"""
Algorithm 6: M-H without φ update
"""

import numpy as np
import time
from .base import MCMCResults
from .base_algorithm import BaseAlgorithm


class Algorithm6(BaseAlgorithm):
    """
    Algorithm 6: Metropolis-Hastings (R=4) without φ update

    Similar to Algorithm 5 but φ_c are not updated after c_i updates
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100, R: int = 4) -> MCMCResults:
        """Run Algorithm 6"""
        c, phi, theta = self.initialize()

        unique_c = np.unique(c)
        c_mapping = {uc: i for i, uc in enumerate(unique_c)}
        c = np.array([c_mapping[ci] for ci in c])
        phi = [phi[uc] for uc in unique_c]

        c_samples = []
        theta_samples = []

        start_time = time.time()

        for iteration in range(n_iter + burn_in):
            # Update each c_i (same as Algorithm 5)
            for i in range(self.n):
                for r in range(R):
                    current_c = c[i]
                    current_phi = phi[current_c]

                    c_temp = np.delete(c, i)
                    n_minus_i_c = np.bincount(c_temp, minlength=len(phi)+1)

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

                    if proposed_c == len(phi):
                        proposed_phi = self.model.sample_prior()
                        is_new = True
                    else:
                        proposed_phi = phi[proposed_c]
                        is_new = False

                    log_accept = (self.model.log_likelihood(self.y[i], proposed_phi) -
                                  self.model.log_likelihood(self.y[i], current_phi))

                    if np.log(np.random.random()) < log_accept:
                        if is_new:
                            phi.append(proposed_phi)
                        c[i] = proposed_c

            # NO φ update (this is the difference from Algorithm 5)

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
