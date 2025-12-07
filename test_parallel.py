"""
Quick test to verify parallel execution works
"""

import numpy as np
import sys
from pathlib import Path
from multiprocessing import cpu_count

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample.visualization import plot_autocorr_time_vs_dimension


def main():
    print("=" * 80)
    print("Testing Parallel Execution")
    print("=" * 80)
    print(f"\nNumber of CPUs available: {cpu_count()}")

    # Small test with just 3 dimensions
    dimensions = [10, 20, 50]

    # Only test 2 fast algorithms
    algorithms = ['Alg3', 'Alg7']

    print(f"\nTesting dimensions: {dimensions}")
    print(f"Testing algorithms: {algorithms}")
    print("\nRunning with parallel execution...")

    iid_times = plot_autocorr_time_vs_dimension(
        dimensions=dimensions,
        algorithms=algorithms,
        n_iter=500,  # Fewer iterations for quick test
        burn_in=50,
        alpha=1.0,
        sigma=0.1,
        mu0=0.0,
        sigma0=1.0,
        autocorr_variable='k',
        results_dir='results',
        save_fig=True,
        random_seed=42,
        n_jobs=-1  # Parallel
    )

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print("\nResults:")
    for algo, times in iid_times.items():
        print(f"{algo}: {times}")

    return iid_times


if __name__ == "__main__":
    result = main()
