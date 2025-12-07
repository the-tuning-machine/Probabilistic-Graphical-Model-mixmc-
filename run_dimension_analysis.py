"""
Analyze how computational cost varies with data dimension
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample.visualization import plot_autocorr_time_vs_dimension


def main():
    """
    Plot time to get 2 i.i.d. samples as a function of data dimension
    """

    print("=" * 80)
    print("Dimension Analysis: Time for 2 i.i.d. samples vs Data Dimension")
    print("=" * 80)

    # Test different dimensions
    dimensions = np.flip(2**np.arange(1, 10))  # 2, 4, 8, 16, 32, 64, 128, 256, 512

    # Select algorithms to compare
    algorithms = ['Alg3', 'Alg5', 'Alg7', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    # Run analysis for autocorrelation on k (number of components)
    print("\n\nAnalyzing autocorrelation on k (number of components)...")
    print("Running in PARALLEL mode (using all available CPUs)")
    iid_times_k = plot_autocorr_time_vs_dimension(
        dimensions=dimensions,
        algorithms=algorithms,
        n_iter=1000,
        burn_in=100,
        alpha=1.0,
        sigma=0.1,
        mu0=0.0,
        sigma0=1.0,
        autocorr_variable='k',
        results_dir='results',
        save_fig=True,
        random_seed=42,
        n_jobs=-1  # Use all CPUs for parallel execution
    )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    # Print summary table
    print("\nTime for 2 i.i.d. samples (ms) - autocorrelation on k:")
    print("-" * 80)
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"n={dim:<8}", end="")
    print()
    print("-" * 80)

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for time in iid_times_k[algo]:
            if np.isnan(time):
                print(f"{'N/A':<13}", end="")
            else:
                print(f"{time:<13.2f}", end="")
        print()

    print("=" * 80)

    return iid_times_k


if __name__ == "__main__":
    iid_times = main()
