import numpy as np
import time
from scipy import stats
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

    def run(self, n_iter: int = 1000, burn_in: int = 100, m: int = 1,
            initial_state: tuple = None) -> MCMCResults:
        """
        Run Algorithm 8

        Args:
            n_iter: Number of iterations
            burn_in: Number of burn-in iterations
            m: Number of auxiliary parameters
            initial_state: Optional tuple (c, phi, theta) for initialization.
                          If None, use default initialization.
        """
        if initial_state is not None:
            c, phi, theta = initial_state
            # Ensure phi is a dict with cluster labels as keys
            if isinstance(phi, (list, np.ndarray)):
                phi = {i: phi[i] for i in range(len(phi)) if phi[i] is not None}
            elif isinstance(phi, dict):
                phi = phi.copy()
            else:
                raise ValueError("phi must be list, array or dict")
        else:
            c, phi_dict, theta = self.initialize()
            # Convert phi to dict
            unique_c = np.unique(c)
            phi = {uc: phi_dict[uc] for uc in unique_c}
        
        n = len(self.y)
        c = c.copy()  # Ensure we work with a copy
        
        # Initialize cluster sizes
        unique_c, cluster_counts = np.unique(c, return_counts=True)
        n_c = {uc: count for uc, count in zip(unique_c, cluster_counts)}
        
        c_samples = []
        theta_samples = []
        phi_samples = []

        start_time = time.time()

        for iteration in range(n_iter + burn_in):
            # PHASE 1: Update assignments c_i (keeping phi fixed)
            for i in range(n):
                # Store old cluster and parameter
                old_c = c[i]
                
                # Remove observation i from its cluster
                n_c[old_c] -= 1
                
                # Check if cluster becomes empty
                if n_c[old_c] == 0:
                    # Remove the cluster
                    del n_c[old_c]
                    reserved_phi = phi[old_c]
                    del phi[old_c]
                    is_singleton = True
                else:
                    is_singleton = False
                
                # List occupied clusters after removal
                # These are clusters with n_c > 0
                occupied_clusters = sorted([k for k in n_c.keys() if n_c[k] > 0])
                k_minus = len(occupied_clusters)
                
                # Get parameters of occupied clusters
                occupied_phi = [phi[k] for k in occupied_clusters]
                
                # Construct m auxiliary candidate parameters
                phi_aux = []
                if is_singleton:
                    # If singleton: keep phi_old as first auxiliary
                    phi_aux.append(reserved_phi)
                    # Draw m-1 new parameters from prior
                    for _ in range(m - 1):
                        phi_aux.append(self.model.sample_prior())
                else:
                    # If not singleton: draw m new parameters
                    for _ in range(m):
                        phi_aux.append(self.model.sample_prior())
                
                # Total candidates: occupied + auxiliary
                h = k_minus + m
                all_candidates = occupied_phi + phi_aux
                
                # Compute LOG probabilities for all candidates
                log_probs = []

                # Existing non-empty components
                for idx in range(k_minus):
                    log_prob = np.log(n_c[occupied_clusters[idx]]) + \
                               self.model.log_likelihood(self.y[i], occupied_phi[idx])
                    log_probs.append(log_prob)

                # Auxiliary components
                b = self.alpha / m
                log_b = np.log(b)
                for aux_idx in range(m):
                    log_prob = log_b + self.model.log_likelihood(self.y[i], phi_aux[aux_idx])
                    log_probs.append(log_prob)

                # Normalize in log-space (subtract max for numerical stability)
                log_probs = np.array(log_probs)
                max_log_prob = np.max(log_probs)
                log_probs_normalized = log_probs - max_log_prob
                probs = np.exp(log_probs_normalized)

                # Handle numerical issues
                if np.isnan(probs).any() or probs.sum() == 0:
                    # Handle numerical underflow
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()

                chosen_idx = np.random.choice(h, p=probs)
                
                # Assign c_i based on chosen component
                if chosen_idx < k_minus:
                    # Assign to existing cluster
                    c[i] = occupied_clusters[chosen_idx]
                    n_c[occupied_clusters[chosen_idx]] += 1
                else:
                    # Create new cluster with the chosen auxiliary parameter
                    # Find new label (max existing + 1)
                    if n_c:
                        new_label = max(n_c.keys()) + 1
                    else:
                        new_label = 0
                    
                    # Add new cluster
                    c[i] = new_label
                    n_c[new_label] = 1
                    phi[new_label] = phi_aux[chosen_idx - k_minus]
            
            # PHASE 2: Update phi for each occupied component (once per iteration)
            occupied_clusters = sorted([k for k in n_c.keys() if n_c[k] > 0])
            
            for cluster_label in occupied_clusters:
                # Get indices of observations in this cluster
                mask = (c == cluster_label)
                if mask.sum() > 0:
                    y_c = self.y[mask]  # Shape (n_c, d)
                    n_c_local = len(y_c)

                    # Update phi using conjugate Gaussian update (diagonal covariance)
                    prec_post = n_c_local / self.model.sigma2 + 1 / self.model.sigma02
                    mu_post = (y_c.sum(axis=0) / self.model.sigma2 + self.model.mu0 / self.model.sigma02) / prec_post
                    sigma_post = 1 / np.sqrt(prec_post)

                    phi[cluster_label] = np.random.normal(mu_post, sigma_post)
            
            # Store samples after burn-in
            if iteration >= burn_in:
                # Relabel for storage to have contiguous labels
                unique_c = np.unique(c)
                label_mapping = {old: new for new, old in enumerate(sorted(unique_c))}
                c_relabeled = np.array([label_mapping[ci] for ci in c])
                
                # Get phi in corresponding order
                phi_relabeled = [phi[old_label] for old_label in sorted(unique_c)]
                
                c_samples.append(c_relabeled.copy())
                theta_samples.append(np.array([phi_relabeled[ci] for ci in c_relabeled]))
                phi_samples.append(phi_relabeled.copy())

        end_time = time.time()

        # Convert samples to arrays
        c_array = np.array(c_samples)
        theta_array = np.array(theta_samples)
        
        # Compute statistics
        k_samples = np.array([len(np.unique(c_sample)) for c_sample in c_samples])
        autocorr_k = self.compute_autocorr(k_samples)
        autocorr_theta1 = self.compute_autocorr(theta_array[:, 0])
        time_per_iter = (end_time - start_time) / n_iter * 1000

        return MCMCResults(
            c=c_array,
            phi=np.array(phi_samples[-1]),  # Last phi
            theta=theta_array,
            time_per_iteration=time_per_iter,
            autocorr_k=autocorr_k,
            autocorr_theta1=autocorr_theta1
        )