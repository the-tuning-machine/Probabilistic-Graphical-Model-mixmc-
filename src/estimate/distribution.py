from abc import ABC, abstractmethod
import numpy as np

class Distribution(ABC):
    """Abstract base class for probability distributions."""
    @abstractmethod
    def __init__(self):
        self.parameters: np.ndarray
        pass

    @abstractmethod
    def pdf(self, x):
        """Calculate the probability density function at point x.

        Args:
            x (float): The point at which to evaluate the PDF.

        Returns:
            float: The value of the PDF at point x.
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int):
        """Generate samples from the distribution.

        Args:
            num_samples (int): The number of samples to generate.
        Returns:
            list: A list of generated samples.
        """
        pass

class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution implementation."""
    def __init__(
            self, 
            mean: np.ndarray | None = None, 
            covariance: np.ndarray | None = None, 
            parameters: np.ndarray | None = None
        ):
        
        assert (mean is not None and covariance is not None) or parameters is not None, \
            "Either mean and covariance or parameters must be provided."

        if parameters is not None:
            dim = int((np.sqrt(8 * len(parameters) + 1) - 1) / 2)
            self.dim = dim
            self.mean = parameters[:dim]
            # Reconstruct covariance from lower triangular elements
            cov_size = dim * (dim + 1) // 2
            cov_flat = parameters[dim:dim + cov_size]
            self.covariance = np.zeros((dim, dim))
            tril_indices = np.tril_indices(dim)
            self.covariance[tril_indices] = cov_flat
            # Make symmetric (copy lower triangle to upper)
            self.covariance = self.covariance + self.covariance.T - np.diag(np.diag(self.covariance))
            self.parameters = parameters
        else:
            assert mean.ndim == 1
            assert covariance.ndim == 2
            assert mean.shape[0] == covariance.shape[0] == covariance.shape[1], \
                "Mean and covariance dimensions must match."
            assert np.allclose(covariance, covariance.T), "Covariance matrix must be symmetric."

            self.mean = mean
            self.covariance = covariance
            self.dim = mean.shape[0]
            self.parameters = np.concatenate([
                self.mean,
                self.covariance[np.tril_indices(self.dim)]
            ])

        self.inv_cov = np.linalg.inv(self.covariance)
        self.norm_const = 1.0 / np.sqrt(
            (2 * np.pi) ** self.dim * np.linalg.det(self.covariance)
        )

    def pdf(self, x: np.ndarray) -> float:
        diff = x - self.mean
        exponent = -0.5 * diff.T @ self.inv_cov @ diff
        return self.norm_const * np.exp(exponent)

    def sample(self, num_samples: int):
        return np.random.multivariate_normal(self.mean, self.covariance, num_samples).tolist()

    def sample_parameters(self, num_samples: int = 1):
        """Generate parameter vectors (mean + covariance) from the prior.

        For a Gaussian prior on theta, we sample:
        - mean from the prior's mean/covariance
        - covariance from an inverse-Wishart (simplified here: just use identity scaled)

        Args:
            num_samples: Number of parameter sets to generate

        Returns:
            List of parameter arrays, each of shape (param_dim,)
        """
        params_list = []
        for _ in range(num_samples):
            # Sample mean from the prior
            sampled_mean = np.random.multivariate_normal(self.mean, self.covariance)

            # For covariance, use a simple fixed covariance scaled by a random factor
            # In a full Bayesian model, this would be inverse-Wishart
            # Here we just use the prior's covariance with some noise
            scale = np.random.gamma(shape=2.0, scale=1.0)  # Random positive scale
            sampled_cov = self.covariance * scale

            # Ensure positive definiteness
            sampled_cov = sampled_cov + np.eye(self.dim) * 0.01

            # Pack into parameter vector
            params = np.concatenate([
                sampled_mean,
                sampled_cov[np.tril_indices(self.dim)]
            ])
            params_list.append(params)

        return params_list

class DirichletDistribution(Distribution):
    """Dirichlet distribution implementation."""
    pass