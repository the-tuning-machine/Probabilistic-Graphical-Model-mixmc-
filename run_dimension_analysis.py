"""
Analyze how computational cost varies with DATA DIMENSION for MCMC algorithms

This script tests the scaling laws by varying the DIMENSION of y_i vectors (d),
not the number of observations (which is fixed at n=20).

Tests:
- Algorithm 1: Basic Gibbs sampling
- Algorithm 8 with m=1, 2, 30: Auxiliary parameter method

Model: Multivariate Normal with diagonal covariance
- F(θ) = N(θ, Σ) with Σ = diag(σ²) and σ² = 0.01
- G₀ = N(0, I) where I is the identity matrix
- α = 1
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from typing import Tuple, Dict, List
import json
import pickle
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample.base import DirichletProcessMixture
from src.sample.algorithm1 import Algorithm1
from src.sample.algorithm8 import Algorithm8


def get_result_filepath(dimension: int, algorithm_name: str, results_dir: str = 'results') -> Path:
    """Get the filepath for saving/loading a specific result"""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    return results_path / f'result_d{dimension:04d}_{algorithm_name}.json'


def save_result(dimension: int, algorithm_name: str, results: Dict, results_dir: str = 'results'):
    """Save a single result to disk"""
    filepath = get_result_filepath(dimension, algorithm_name, results_dir)

    # Add metadata
    save_data = {
        'dimension': dimension,
        'algorithm': algorithm_name,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)


def load_result(dimension: int, algorithm_name: str, results_dir: str = 'results') -> Dict:
    """Load a single result from disk, returns None if not found"""
    filepath = get_result_filepath(dimension, algorithm_name, results_dir)

    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['results']
    except Exception as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None


def check_existing_results(dimensions: List[int], algorithms: List[str],
                          results_dir: str = 'results') -> Tuple[List, List]:
    """
    Check which results already exist

    Returns:
        (existing_jobs, missing_jobs): Lists of (dimension, algorithm, m_val) tuples
    """
    existing = []
    missing = []

    for dim in dimensions:
        for algo in algorithms:
            if algo == 'Alg1':
                m_val = None
            elif algo == 'Alg8_m1':
                m_val = 1
            elif algo == 'Alg8_m2':
                m_val = 2
            elif algo == 'Alg8_m30':
                m_val = 30
            else:
                continue

            result = load_result(dim, algo, results_dir)
            if result is not None:
                existing.append((dim, algo, m_val, result))
            else:
                missing.append((dim, algo, m_val))

    return existing, missing


def generate_multivariate_data(n: int, d: int, seed: int = 42):
    """
    Generate synthetic multivariate data

    Args:
        n: Number of observations
        d: Dimension of each observation
        seed: Random seed

    Returns:
        data: Array of shape (n, d)
    """
    np.random.seed(seed)

    # Create a mixture of two Gaussians in d dimensions
    n_half = n // 2

    # Cluster 1: centered at -1 in all dimensions
    cluster1 = np.random.normal(-1.0, 0.3, size=(n_half, d))

    # Cluster 2: centered at +0.5 in all dimensions
    cluster2 = np.random.normal(0.5, 0.3, size=(n - n_half, d))

    data = np.vstack([cluster1, cluster2])
    # Shuffle
    perm = np.random.permutation(n)
    return data[perm]


def run_single_algorithm(args: Tuple) -> Tuple[int, str, Dict]:
    """
    Run a single algorithm on a single dimension (worker function for parallel execution)

    Args:
        args: Tuple of (dimension, algorithm_name, m_val, n_obs, n_iter, burn_in, results_dir)

    Returns:
        (dimension, algorithm_name, results_dict)
    """
    dimension, algorithm_name, m_val, n_obs, n_iter, burn_in, results_dir = args

    # Generate data with dimension-specific seed
    data = generate_multivariate_data(n_obs, dimension, seed=42 + dimension)

    # Model parameters
    alpha = 1.0
    sigma = 0.1
    mu0 = 0.0
    sigma0 = 1.0

    # Create model
    model = DirichletProcessMixture(data, alpha=alpha, sigma=sigma,
                                    mu0=mu0, sigma0=sigma0)

    # Run algorithm
    if algorithm_name == 'Alg1':
        alg = Algorithm1(model)
        res = alg.run(n_iter=n_iter, burn_in=burn_in)
    elif algorithm_name.startswith('Alg8'):
        alg = Algorithm8(model)
        res = alg.run(n_iter=n_iter, burn_in=burn_in, m=m_val)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Compute results
    results = {
        'autocorr_k': res.autocorr_k,
        'autocorr_theta1': res.autocorr_theta1,
        'time_per_iter': res.time_per_iteration,
    }

    # Save result immediately
    save_result(dimension, algorithm_name, results, results_dir)

    print(f"  ✓ d={dimension:4d} {algorithm_name:10s}: "
          f"τ_k={res.autocorr_k:6.2f}, τ_θ={res.autocorr_theta1:6.2f}, "
          f"time={res.time_per_iteration:6.2f}ms [SAVED]")

    return (dimension, algorithm_name, results)


def run_all_experiments_parallel(dimensions, n_obs=20, n_iter=2000, burn_in=200,
                                n_jobs=-1, results_dir='results'):
    """
    Run all algorithms for all dimensions in parallel (with caching)

    Args:
        dimensions: List of dimensions to test
        n_obs: Number of observations (fixed)
        n_iter: Number of MCMC iterations
        burn_in: Burn-in period
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        results_dir: Directory to save/load results

    Returns:
        all_results: List of dictionaries, one per dimension
    """
    algorithms = ['Alg1', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    # Check which results already exist
    existing, missing = check_existing_results(dimensions, algorithms, results_dir)

    print(f"\n{'='*80}")
    print(f"CHECKING CACHED RESULTS")
    print(f"{'='*80}")
    print(f"Total jobs: {len(dimensions)} dimensions × {len(algorithms)} algorithms = {len(dimensions) * len(algorithms)} jobs")
    print(f"✓ Found {len(existing)} cached results")
    print(f"⚠ Missing {len(missing)} results")

    if existing:
        print(f"\nLoaded results:")
        for dim, algo, _, _ in sorted(existing)[:10]:  # Show first 10
            print(f"  ✓ d={dim:4d} {algo:10s}")
        if len(existing) > 10:
            print(f"  ... and {len(existing) - 10} more")

    # Organize existing results by dimension
    results_by_dim = {dim: {} for dim in dimensions}
    for dim, algo_name, m_val, result in existing:
        results_by_dim[dim][algo_name] = result

    # If there are missing results, compute them
    if missing:
        # Determine number of processes
        if n_jobs == -1:
            n_processes = cpu_count()
        else:
            n_processes = min(n_jobs, cpu_count())

        print(f"\n{'='*80}")
        print(f"PARALLEL EXECUTION")
        print(f"{'='*80}")
        print(f"Using {n_processes} CPU cores")
        print(f"Computing {len(missing)} missing jobs...")
        print()

        # Create jobs list with results_dir
        jobs = []
        for dim, algo_name, m_val in missing:
            jobs.append((dim, algo_name, m_val, n_obs, n_iter, burn_in, results_dir))

        # Run missing jobs in parallel
        print("Running experiments in parallel...")
        print()
        with Pool(processes=n_processes) as pool:
            raw_results = pool.map(run_single_algorithm, jobs)

        print()
        print("All parallel jobs completed!")

        # Add new results
        for dim, algo_name, results in raw_results:
            results_by_dim[dim][algo_name] = results
    else:
        print("\n✓ All results already cached! No computation needed.")

    # Convert to list format
    all_results = [results_by_dim[dim] for dim in dimensions]

    return all_results


def plot_results(all_results, dimensions):
    """
    Plot three separate metrics vs dimension

    Args:
        all_results: List of result dicts, one per dimension
        dimensions: List of dimensions tested
    """
    algorithms = ['Alg1', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    # Extract data for each algorithm
    time_per_iter = {algo: [] for algo in algorithms}
    autocorr_k = {algo: [] for algo in algorithms}
    autocorr_theta1 = {algo: [] for algo in algorithms}

    for results in all_results:
        for algo in algorithms:
            time_per_iter[algo].append(results[algo]['time_per_iter'])
            autocorr_k[algo].append(results[algo]['autocorr_k'])
            autocorr_theta1[algo].append(results[algo]['autocorr_theta1'])

    # Create three plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Time per iteration
    ax = axes[0]
    for algo in algorithms:
        ax.plot(dimensions, time_per_iter[algo], marker='o',
                label=algo, linewidth=2, markersize=6)
    ax.set_xlabel('Data Dimension (d)', fontsize=12)
    ax.set_ylabel('Time per Iteration (ms)', fontsize=12)
    ax.set_title('Computational Cost vs Dimension',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot 2: Autocorrelation for k
    ax = axes[1]
    for algo in algorithms:
        ax.plot(dimensions, autocorr_k[algo], marker='s',
                label=algo, linewidth=2, markersize=6)
    ax.set_xlabel('Data Dimension (d)', fontsize=12)
    ax.set_ylabel('Autocorrelation Time τ_k', fontsize=12)
    ax.set_title('Autocorrelation Time vs Dimension\n(Number of components k)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot 3: Autocorrelation for theta_1
    ax = axes[2]
    for algo in algorithms:
        ax.plot(dimensions, autocorr_theta1[algo], marker='^',
                label=algo, linewidth=2, markersize=6)
    ax.set_xlabel('Data Dimension (d)', fontsize=12)
    ax.set_ylabel('Autocorrelation Time τ_θ₁', fontsize=12)
    ax.set_title('Autocorrelation Time vs Dimension\n(First parameter θ₁)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    output_path = Path('results') / 'scaling_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to {output_path}")

    plt.close()


def print_summary_table(all_results, dimensions):
    """Print summary tables for all three metrics"""
    algorithms = ['Alg1', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    # Table 1: Time per iteration
    print("\n" + "=" * 100)
    print("Time per Iteration (ms)")
    print("=" * 100)
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"d={dim:<10}", end="")
    print()
    print("-" * 100)

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for results in all_results:
            val = results[algo]['time_per_iter']
            print(f"{val:<10.2f}", end="")
        print()

    # Table 2: Autocorrelation time τ_k
    print("\n" + "=" * 100)
    print("Autocorrelation Time τ_k (integrated autocorrelation for k)")
    print("=" * 100)
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"d={dim:<10}", end="")
    print()
    print("-" * 100)

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for results in all_results:
            val = results[algo]['autocorr_k']
            print(f"{val:<10.2f}", end="")
        print()

    # Table 3: Autocorrelation time τ_θ₁
    print("\n" + "=" * 100)
    print("Autocorrelation Time τ_θ₁ (integrated autocorrelation for θ₁)")
    print("=" * 100)
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"d={dim:<10}", end="")
    print()
    print("-" * 100)

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for results in all_results:
            val = results[algo]['autocorr_theta1']
            print(f"{val:<10.2f}", end="")
        print()


def main():
    """Main experiment"""
    print("=" * 80)
    print("DIMENSION SCALING ANALYSIS - Algorithm 1 and Algorithm 8")
    print("=" * 80)
    print()
    print("Testing multivariate Gaussian model:")
    print("  - F(θ) = N(θ, Σ) with Σ = diag(0.01)")
    print("  - G₀ = N(0, I)")
    print("  - α = 1")
    print("  - Number of observations: n = 20 (fixed)")
    print("  - Varying dimension: d = 2, 4, 8, ..., 1024")
    print()

    # Test different dimensions
    dimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"Dimensions to test: {dimensions}")
    print()

    # MCMC parameters
    n_iter = 20000
    burn_in = 200
    print(f"MCMC parameters:")
    print(f"  - Iterations: {n_iter}")
    print(f"  - Burn-in: {burn_in}")

    # Run experiments in parallel (with automatic caching)
    all_results = run_all_experiments_parallel(
        dimensions=dimensions,
        n_obs=20,
        n_iter=n_iter,
        burn_in=burn_in,
        n_jobs=-1,  # Use all available CPUs
        results_dir='results'
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    print_summary_table(all_results, dimensions)

    # Plot results
    plot_results(all_results, dimensions)

if __name__ == "__main__":
    main()
