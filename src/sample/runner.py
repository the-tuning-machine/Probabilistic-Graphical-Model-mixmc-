"""
Runner for MCMC algorithms
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List
from .base import DirichletProcessMixture, MCMCResults
from .algorithm1 import Algorithm1
from .algorithm5 import Algorithm5
from .algorithm8 import Algorithm8


class MCMCRunner:
    """Orchestrator for running MCMC algorithms"""

    ALGORITHMS = {
        'Alg1': Algorithm1,
        'Alg5': Algorithm5,
        'Alg8_m1': (Algorithm8, {'m': 1}),
        'Alg8_m2': (Algorithm8, {'m': 2}),
        'Alg8_m30': (Algorithm8, {'m': 30}),
    }

    def __init__(self, data: np.ndarray, alpha: float = 1.0,
                 sigma: float = 0.1, mu0: float = 0.0, sigma0: float = 1.0,
                 results_dir: str = 'results'):
        """
        Initialize the MCMC runner

        Args:
            data: Observed data
            alpha: Concentration parameter
            sigma: Standard deviation of likelihood
            mu0: Mean of prior
            sigma0: Standard deviation of prior
            results_dir: Directory to save results
        """
        self.model = DirichletProcessMixture(data, alpha, sigma, mu0, sigma0)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, MCMCResults] = {}

    def run_algorithm(self, algorithm_name: str, n_iter: int = 1000,
                     burn_in: int = 100, **kwargs) -> MCMCResults:
        """
        Run a specific algorithm

        Args:
            algorithm_name: Name of the algorithm (e.g., 'Alg1', 'Alg8_m1')
            n_iter: Number of iterations
            burn_in: Number of burn-in iterations
            **kwargs: Additional algorithm-specific parameters

        Returns:
            MCMCResults object
        """
        if algorithm_name not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                           f"Available: {list(self.ALGORITHMS.keys())}")

        algo_spec = self.ALGORITHMS[algorithm_name]

        # Handle tuple specification (Algorithm class, default params)
        if isinstance(algo_spec, tuple):
            AlgorithmClass, default_params = algo_spec
            params = {**default_params, **kwargs}
        else:
            AlgorithmClass = algo_spec
            params = kwargs

        print(f"Running {algorithm_name}...")
        algorithm = AlgorithmClass(self.model)
        results = algorithm.run(n_iter=n_iter, burn_in=burn_in, **params)

        self.results[algorithm_name] = results
        return results

    def run_all(self, n_iter: int = 1000, burn_in: int = 100,
                algorithms: Optional[List[str]] = None) -> Dict[str, MCMCResults]:
        """
        Run all or selected algorithms

        Args:
            n_iter: Number of iterations
            burn_in: Number of burn-in iterations
            algorithms: List of algorithm names to run (None = all)

        Returns:
            Dictionary of results
        """
        if algorithms is None:
            algorithms = list(self.ALGORITHMS.keys())

        for algo_name in algorithms:
            self.run_algorithm(algo_name, n_iter, burn_in)

        return self.results

    def save_results(self, filename: str = 'mcmc_results.pkl'):
        """
        Save results to disk

        Args:
            filename: Name of the file to save
        """
        filepath = self.results_dir / filename

        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)

        print(f"Results saved to {filepath}")

        # Also save summary as JSON
        summary = self.get_summary()
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {json_path}")

    def load_results(self, filename: str = 'mcmc_results.pkl'):
        """
        Load results from disk

        Args:
            filename: Name of the file to load
        """
        filepath = self.results_dir / filename

        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)

        print(f"Results loaded from {filepath}")
        return self.results

    def get_summary(self) -> Dict:
        """
        Get summary statistics for all results

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        for name, res in self.results.items():
            summary[name] = {
                'time_per_iteration': float(res.time_per_iteration),
                'autocorr_k': float(res.autocorr_k),
                'autocorr_theta1': float(res.autocorr_theta1),
                'n_samples': len(res.c),
                'final_n_components': int(len(np.unique(res.c[-1]))),
            }

        return summary

    def print_summary(self):
        """Print summary table"""
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Algorithm':<35} {'Time/iter':<15} {'Autocorr k':<15} {'Autocorr θ₁':<15}")
        print("-" * 80)

        for name, res in self.results.items():
            time_str = f"{res.time_per_iteration:.1f} ms"
            autocorr_k_str = f"{res.autocorr_k:.1f}"
            autocorr_theta_str = f"{res.autocorr_theta1:.1f}"

            print(f"{name:<35} {time_str:<15} {autocorr_k_str:<15} {autocorr_theta_str:<15}")

        print("=" * 80)

    def compute_iid_time(self, algorithm_name: str, autocorr_variable: str = 'k') -> float:
        """
        Compute time to get 2 independent samples

        Args:
            algorithm_name: Name of the algorithm
            autocorr_variable: 'k' or 'theta1'

        Returns:
            Time in milliseconds to get 2 i.i.d. samples
        """
        if algorithm_name not in self.results:
            raise ValueError(f"Algorithm {algorithm_name} has not been run yet")

        res = self.results[algorithm_name]

        if autocorr_variable == 'k':
            rho = res.autocorr_k
        elif autocorr_variable == 'theta1':
            rho = res.autocorr_theta1
        else:
            raise ValueError(f"Unknown variable: {autocorr_variable}")

        # Autocorrelation time: tau = 1 / (1 - rho)
        # Number of iterations needed for independent samples: n_iter = tau
        tau = 1 / (1 - rho) if rho < 1 else float('inf')

        # Time for 2 independent samples
        time_for_2_iid = 2 * tau * res.time_per_iteration

        return time_for_2_iid
