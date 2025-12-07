import numpy as np
from typing import Tuple, List
from multiprocessing import Pool, cpu_count
from functools import partial
from src.estimate.distribution import Distribution, NormalDistribution


def generate_dirichlet_process_sample(
    n_data_point: int,
    alpha: float,
    G_0: Distribution
) -> Tuple[List[Distribution], np.ndarray, np.ndarray]:
    """Generate one sample using Chinese Restaurant Process.

    For a single sample, generates n_data_point theta values using CRP,
    then for each theta generates one y observation.

    Uses the sequential representation:
    theta_i | theta_1, ..., theta_{i-1} ~
        (1/(alpha + i - 1)) * sum_{j=1}^{i-1} delta(theta_j) +
        (alpha/(alpha + i - 1)) * G_0

    Args:
        n_data_point: Number of data points (theta, y pairs) in this sample
        alpha: Concentration parameter of the DP
        G_0: Base distribution

    Returns:
        theta_distributions: List of distributions theta_i (one per data point)
        y_array: Observations y_i of shape (n_data_point, obs_dim)
        theta_params: Parameters of theta_i of shape (n_data_point, param_dim)
    """
    # Lists to store unique atoms (clusters) and their counts
    unique_atoms = []  # List of unique theta distributions
    atom_counts = []   # Number of times each atom has been selected

    # Lists for all data points
    theta_distributions = []
    theta_params = []
    y_list = []

    for i in range(n_data_point):
        # Compute probabilities for Chinese Restaurant Process
        total = alpha + i
        probs = []

        # Probability of joining existing clusters
        for count in atom_counts:
            probs.append(count / total)

        # Probability of creating new cluster
        probs.append(alpha / total)

        # Sample which cluster to join (or create new)
        cluster_choice = np.random.choice(len(probs), p=probs)

        if cluster_choice < len(unique_atoms):
            # Join existing cluster
            theta_i = unique_atoms[cluster_choice]
            atom_counts[cluster_choice] += 1
        else:
            # Create new cluster: sample from G_0
            if isinstance(G_0, NormalDistribution):
                theta_params_new = G_0.sample_parameters(1)[0]
                theta_i = NormalDistribution(parameters=theta_params_new)
            else:
                raise NotImplementedError(f"Distribution type {type(G_0)} not supported")

            unique_atoms.append(theta_i)
            atom_counts.append(1)

        # Store theta_i
        theta_distributions.append(theta_i)
        theta_params.append(theta_i.parameters)

        # Sample y_i ~ P(y | theta_i)
        y_i = np.array(theta_i.sample(1)[0])
        y_list.append(y_i)

    return theta_distributions, np.array(y_list), np.array(theta_params)


def _generate_single_sample(args):
    """Wrapper function for parallel processing.

    Args:
        args: Tuple of (sample_index, n_data_point, alpha, G_0, base_seed)

    Returns:
        Tuple of (y_sample, theta_sample)
    """
    sample_idx, n_data_point, alpha, G_0, base_seed = args
    # Set a unique random seed for each sample
    np.random.seed(base_seed + sample_idx)
    _, y_sample, theta_sample = generate_dirichlet_process_sample(
        n_data_point=n_data_point,
        alpha=alpha,
        G_0=G_0
    )
    return y_sample, theta_sample


class DiffusionDataset:
    """Dataset for training the diffusion model.

    Each sample contains:
    - y: observations of shape (n_data_point, obs_dim)
    - theta: distribution parameters of shape (n_data_point, param_dim)
    """

    def __init__(
        self,
        n_sample: int,
        n_data_point: int,
        alpha: float,
        G_0: Distribution,
        n_workers: int = None
    ):
        """
        Args:
            n_sample: Number of samples in the dataset
            n_data_point: Number of (theta, y) pairs per sample
            alpha: DP concentration parameter
            G_0: Base distribution
            n_workers: Number of parallel workers (default: cpu_count())
        """
        self.n_sample = n_sample
        self.n_data_point = n_data_point
        self.alpha = alpha
        self.G_0 = G_0

        # Determine number of workers
        if n_workers is None:
            n_workers = cpu_count()

        print(f"Using {n_workers} parallel workers for dataset generation...")

        # Generate all samples using Chinese Restaurant Process (parallelized)
        # Create arguments for each sample
        # Use a base seed to ensure reproducibility while maintaining parallelism
        base_seed = np.random.randint(0, 1000000)
        args_list = [
            (i, n_data_point, alpha, G_0, base_seed)
            for i in range(n_sample)
        ]

        # Use multiprocessing Pool to parallelize
        with Pool(processes=n_workers) as pool:
            results = pool.map(_generate_single_sample, args_list)

        # Unpack results
        all_y = [y for y, _ in results]
        all_theta = [theta for _, theta in results]

        # Shape: (n_sample, n_data_point, obs_dim/param_dim)
        self.y_samples = np.array(all_y)
        self.theta_params = np.array(all_theta)

        self.obs_dim = self.y_samples.shape[2]
        self.param_dim = self.theta_params.shape[2]

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        """Return (y, theta) for the given index.

        Returns:
            y: shape (n_data_point, obs_dim)
            theta: shape (n_data_point, param_dim)
        """
        return self.y_samples[idx], self.theta_params[idx]

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a random batch of data.

        Returns:
            y_batch: (batch_size, n_data_point, obs_dim)
            theta_batch: (batch_size, n_data_point, param_dim)
        """
        indices = np.random.choice(self.n_sample, size=batch_size, replace=False)
        return self.y_samples[indices], self.theta_params[indices]
