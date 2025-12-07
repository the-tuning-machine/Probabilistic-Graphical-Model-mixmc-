"""
Algorithm 8: Auxiliary Parameter Method
"""

import numpy as np
import time
from .base import MCMCResults
from .base_algorithm import BaseAlgorithm


class Algorithm8(BaseAlgorithm):
    """
    Algorithm 8: Gibbs sampling with m auxiliary parameters

    At each iteration:
    1. For i = 1,...,n:
       - Draw m auxiliary parameters φ_{-i,1},...,φ_{-i,m} from G_0
       - Sample c_i from full conditional given auxiliary parameters
    2. Update φ_c for each component c
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100, m: int = 1) -> MCMCResults:
        """Run Algorithm 8"""
        c, phi, theta = self.initialize()

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
                # Draw m auxiliary parameters
                phi_aux = [self.model.sample_prior() for _ in range(m)]

                # Count components excluding i
                c_temp = np.delete(c, i)
                n_minus_i_c = np.bincount(c_temp, minlength=k)

                # Compute probabilities
                probs = []
                labels = []  # Will store either existing component index or ('aux', aux_index)

                # Existing non-empty components (excluding i)
                for j in range(k):
                    if n_minus_i_c[j] > 0:
                        prob = n_minus_i_c[j] * self.model.likelihood(self.y[i], phi[j])
                        probs.append(prob)
                        labels.append(('existing', j))

                # Auxiliary components
                b = self.alpha / m
                for aux_idx, phi_a in enumerate(phi_aux):
                    prob = b * self.model.likelihood(self.y[i], phi_a)
                    probs.append(prob)
                    labels.append(('auxiliary', aux_idx))

                # Sample
                probs = np.array(probs)
                probs /= probs.sum()
                chosen_idx = np.random.choice(len(probs), p=probs)

                # Assign c_i based on chosen component
                label_type, label_idx = labels[chosen_idx]
                if label_type == 'existing':
                    c[i] = label_idx
                else:  # auxiliary
                    # Create new component with the chosen auxiliary parameter
                    phi.append(phi_aux[label_idx])
                    c[i] = len(phi) - 1

            # Update phi for each occupied component
            for j in range(len(phi)):
                mask = (c == j)
                if mask.sum() > 0:
                    y_c = self.y[mask]
                    n_c = len(y_c)
                    prec_post = n_c / self.model.sigma2 + 1 / self.model.sigma02
                    mu_post = (y_c.sum() / self.model.sigma2 + self.model.mu0 / self.model.sigma02) / prec_post
                    sigma_post = 1 / np.sqrt(prec_post)
                    phi[j] = np.random.normal(mu_post, sigma_post)

            # Remove empty components and relabel
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
