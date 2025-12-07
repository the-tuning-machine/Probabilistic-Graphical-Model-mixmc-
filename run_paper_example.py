"""
Reproduce the example from Neal (1998) Table 1
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample.runner import MCMCRunner
from src.sample.visualization import plot_comparison_table1


def main():
    """Run all 8 algorithms and compare results with Table 1"""

    # Data from the paper (page 14-15)
    data = np.array([-1.48, -1.40, -1.16, -1.08, -1.02,
                     0.14, 0.51, 0.53, 0.78])

    # Model parameters from the paper
    alpha = 1.0
    sigma = 0.1  # σ² = 0.01
    mu0 = 0.0
    sigma0 = 1.0

    # Number of iterations
    n_iter = 1000
    burn_in = 100

    print("=" * 80)
    print("Dirichlet Process Mixture Model - MCMC Algorithms")
    print("Based on Neal (1998)")
    print("=" * 80)
    print(f"\nData: {data}")
    print(f"n = {len(data)}, α = {alpha}, σ = {sigma}, σ₀ = {sigma0}")
    print(f"\nRunning {n_iter} iterations with {burn_in} burn-in")
    print("=" * 80)

    # Create runner
    runner = MCMCRunner(data, alpha, sigma, mu0, sigma0, results_dir='results')

    # Run all algorithms
    algorithms = [
        'Alg1', 'Alg2', 'Alg3', 'Alg4', 'Alg5',
        'Alg6', 'Alg7', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30'
    ]

    runner.run_all(n_iter, burn_in, algorithms)

    # Print summary
    print("\n")
    runner.print_summary()

    # Compare with Table 1
    print("\nComparison with Table 1 from Neal (1998):")
    print("-" * 80)

    expected = {
        'Alg4': (7.6, 13.7, 8.5),
        'Alg5': (8.6, 8.1, 10.2),
        'Alg6': (8.3, 19.4, 64.1),
        'Alg7': (8.0, 6.9, 5.3),
        'Alg8_m1': (7.9, 5.2, 5.6),
        'Alg8_m2': (8.8, 3.7, 4.7),
        'Alg8_m30': (38.0, 2.0, 2.8),
    }

    for name, res in runner.results.items():
        if name in expected:
            exp_time, exp_k, exp_theta = expected[name]
            print(f"\n{name}:")
            print(f"  Observed: time={res.time_per_iteration:.1f}ms, "
                  f"autocorr_k={res.autocorr_k:.1f}, autocorr_θ₁={res.autocorr_theta1:.1f}")
            print(f"  Expected: time={exp_time:.1f}ms, "
                  f"autocorr_k={exp_k:.1f}, autocorr_θ₁={exp_theta:.1f}")

    print("\n" + "=" * 80)
    print("Note: Exact values may differ due to randomness and implementation details,")
    print("but trends should be similar to Table 1 in the paper.")
    print("=" * 80)

    # Save results
    runner.save_results('paper_example_results.pkl')

    # Create comparison plot
    plot_comparison_table1(runner.results, save_fig=True, results_dir='results')

    return runner


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    runner = main()
