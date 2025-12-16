"""
Visualization tools for MCMC results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from .runner import MCMCRunner


def _process_single_algorithm(args: Tuple) -> Tuple[str, float, object]:
    """
    Process a single algorithm for a given dimension (helper function for parallel execution)

    Args:
        args: Tuple of (algo_name, data, alpha, sigma, mu0, sigma0,
                       n_iter, burn_in, autocorr_variable, results_dir)

    Returns:
        Tuple of (algorithm_name, time_2_iid, mcmc_results)
    """
    (algo_name, data, alpha, sigma, mu0, sigma0,
     n_iter, burn_in, autocorr_variable, results_dir) = args

    try:
        # Create runner for this algorithm
        runner = MCMCRunner(data, alpha, sigma, mu0, sigma0, results_dir)

        # Run only this algorithm
        results = runner.run_algorithm(algo_name, n_iter, burn_in)

        # Compute time for 2 i.i.d. samples
        time_2_iid = runner.compute_iid_time(algo_name, autocorr_variable)

        print(f"  {algo_name}: {time_2_iid:.2f} ms for 2 i.i.d. samples")

        return algo_name, time_2_iid, results

    except Exception as e:
        print(f"  Error with {algo_name}: {e}")
        return algo_name, np.nan, None


def plot_autocorr_time_vs_dimension(
    dimensions: List[int],
    algorithms: Optional[List[str]] = None,
    n_iter: int = 1000,
    burn_in: int = 100,
    alpha: float = 1.0,
    sigma: float = 0.1,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    autocorr_variable: str = 'k',
    results_dir: str = 'results',
    save_fig: bool = True,
    random_seed: int = 42,
    n_jobs: int = -1
):
    """
    Plot time to get 2 i.i.d. samples vs data dimension (with parallel execution)

    Args:
        dimensions: List of data dimensions to test
        algorithms: List of algorithm names to compare
        n_iter: Number of MCMC iterations
        burn_in: Number of burn-in iterations
        alpha: Concentration parameter
        sigma: Likelihood std dev
        mu0: Prior mean
        sigma0: Prior std dev
        autocorr_variable: 'k' or 'theta1'
        results_dir: Directory to save results
        save_fig: Whether to save the figure
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
    """
    if algorithms is None:
        algorithms = ['Alg3', 'Alg5', 'Alg7', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    # Determine number of processes
    if n_jobs == -1:
        n_processes = cpu_count()
    elif n_jobs == 1:
        n_processes = 1
    else:
        n_processes = min(n_jobs, cpu_count())

    print(f"\nUsing {n_processes} parallel processes (algorithms in parallel, dimensions sequential)")
    print(f"Processing {len(dimensions)} dimensions × {len(algorithms)} algorithms")

    # Store results
    iid_times = {algo: [] for algo in algorithms}

    # Process each dimension SEQUENTIALLY
    for dim_idx, dim in enumerate(dimensions):
        print(f"\n{'='*80}")
        print(f"[{dim_idx+1}/{len(dimensions)}] Processing dimension: {dim}")
        print(f"{'='*80}")

        # Set seed for this dimension
        np.random.seed(random_seed + dim)

        # Generate synthetic data
        n_half = dim // 2
        data = np.concatenate([
            np.random.normal(-1.0, 0.3, n_half),
            np.random.normal(0.5, 0.3, dim - n_half)
        ])
        np.random.shuffle(data)

        # Prepare arguments for each algorithm (to run in parallel)
        args_list = [
            (algo, data, alpha, sigma, mu0, sigma0,
             n_iter, burn_in, autocorr_variable, results_dir)
            for algo in algorithms
        ]

        # Run algorithms IN PARALLEL for this dimension
        if n_processes > 1:
            with Pool(processes=n_processes) as pool:
                algo_results = pool.map(_process_single_algorithm, args_list)
        else:
            algo_results = [_process_single_algorithm(args) for args in args_list]

        # Store results for this dimension
        print(f"\n  Saving results for dimension {dim}...")
        runner = MCMCRunner(data, alpha, sigma, mu0, sigma0, results_dir)

        for algo_name, time_2_iid, mcmc_results in algo_results:
            iid_times[algo_name].append(time_2_iid)

            # Store MCMC results in runner for saving
            if mcmc_results is not None:
                runner.results[algo_name] = mcmc_results

        # Save all results for this dimension
        runner.save_results(f'results_dim{dim}.pkl')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algorithms:
        times = np.array(iid_times[algo])
        valid = ~np.isnan(times)
        if valid.any():
            ax.plot(np.array(dimensions)[valid], times[valid],
                   marker='o', label=algo, linewidth=2, markersize=6)

    ax.set_xlabel('Data Dimension (n)', fontsize=12)
    ax.set_ylabel('Time for 2 i.i.d. samples (ms)', fontsize=12)
    ax.set_title(f'Computational Cost vs Dimension (autocorr on {autocorr_variable})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()

    if save_fig:
        fig_path = Path(results_dir) / 'iid_time_vs_dimension.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {fig_path}")

    plt.show()

    return iid_times


def plot_comparison_table1(results: Dict, save_fig: bool = True, results_dir: str = 'results'):
    """
    Plot comparison with Table 1 from Neal (1998)

    Args:
        results: Dictionary of MCMCResults
        save_fig: Whether to save the figure
        results_dir: Directory to save results
    """
    # Expected values from Table 1
    expected = {
        'Alg4': (7.6, 13.7, 8.5),
        'Alg5': (8.6, 8.1, 10.2),
        'Alg6': (8.3, 19.4, 64.1),
        'Alg7': (8.0, 6.9, 5.3),
        'Alg8_m1': (7.9, 5.2, 5.6),
        'Alg8_m2': (8.8, 3.7, 4.7),
        'Alg8_m30': (38.0, 2.0, 2.8),
    }

    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['time_per_iteration', 'autocorr_k', 'autocorr_theta1']
    titles = ['Time per Iteration (ms)', 'Autocorrelation (k)', 'Autocorrelation (θ₁)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        algos = []
        observed = []
        expected_vals = []

        for name, res in results.items():
            if name in expected:
                algos.append(name)

                if metric == 'time_per_iteration':
                    observed.append(res.time_per_iteration)
                    expected_vals.append(expected[name][0])
                elif metric == 'autocorr_k':
                    observed.append(res.autocorr_k)
                    expected_vals.append(expected[name][1])
                else:  # autocorr_theta1
                    observed.append(res.autocorr_theta1)
                    expected_vals.append(expected[name][2])

        x = np.arange(len(algos))
        width = 0.35

        ax.bar(x - width/2, observed, width, label='Observed', alpha=0.8)
        ax.bar(x + width/2, expected_vals, width, label='Expected (Table 1)', alpha=0.8)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_fig:
        fig_path = Path(results_dir) / 'comparison_table1.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {fig_path}")

    plt.show()


def plot_trace_plots(results: Dict, algorithm_name: str,
                     save_fig: bool = True, results_dir: str = 'results'):
    """
    Plot trace plots for number of components and theta_1

    Args:
        results: Dictionary of MCMCResults
        algorithm_name: Name of algorithm to plot
        save_fig: Whether to save the figure
        results_dir: Directory to save results
    """
    if algorithm_name not in results:
        raise ValueError(f"Algorithm {algorithm_name} not found in results")

    res = results[algorithm_name]
    k_samples = np.array([len(np.unique(c)) for c in res.c])
    theta1_samples = res.theta[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot k
    axes[0].plot(k_samples, linewidth=0.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Number of Components (k)')
    axes[0].set_title(f'{algorithm_name}: Number of Components')
    axes[0].grid(True, alpha=0.3)

    # Plot theta_1
    axes[1].plot(theta1_samples, linewidth=0.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('θ₁')
    axes[1].set_title(f'{algorithm_name}: First Parameter θ₁')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        fig_path = Path(results_dir) / f'trace_{algorithm_name}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Trace plot saved to {fig_path}")

    plt.show()
