import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from typing import Tuple, Dict, List
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.sample.base import DirichletProcessMixture
from src.sample.algorithm1 import Algorithm1
from src.sample.algorithm8 import Algorithm8
from src.estimate.data import generate_dirichlet_process_sample
from src.estimate.distribution import NormalDistribution


def get_result_filepath(dimension: int, algorithm_name: str, results_dir: str = 'results') -> Path:
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    return results_path / f'result_d{dimension:04d}_{algorithm_name}.json'


def save_result(dimension: int, algorithm_name: str, results: Dict, results_dir: str = 'results'):
    filepath = get_result_filepath(dimension, algorithm_name, results_dir)

    save_data = {
        'dimension': dimension,
        'algorithm': algorithm_name,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)


def load_result(dimension: int, algorithm_name: str, results_dir: str = 'results') -> Dict:
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


def generate_multivariate_data(alpha: float, n: int, d: int, seed: int = 42):
    np.random.seed(seed)

    mean_0 = np.zeros(d)
    cov_0 = np.eye(d)
    G_0 = NormalDistribution(mean=mean_0, covariance=cov_0)

    theta_distributions, y_array, theta_params = generate_dirichlet_process_sample(
        n_data_point=n,
        alpha=alpha,
        G_0=G_0
    )

    return y_array


def run_single_algorithm(args: Tuple) -> Tuple[int, str, Dict]:
    dimension, algorithm_name, m_val, n_obs, n_iter, burn_in, results_dir = args

    alpha = 1.0
    sigma = 0.1
    mu0 = 0.0
    sigma0 = 1.0

    data = generate_multivariate_data(alpha, n_obs, dimension, seed=42 + dimension)
    model = DirichletProcessMixture(data, alpha=alpha, sigma=sigma,
                                    mu0=mu0, sigma0=sigma0)
    if algorithm_name == 'Alg1':
        alg = Algorithm1(model)
        res = alg.run(n_iter=n_iter, burn_in=burn_in)
    elif algorithm_name.startswith('Alg8'):
        alg = Algorithm8(model)
        res = alg.run(n_iter=n_iter, burn_in=burn_in, m=m_val)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    results = {
        'autocorr_k': res.autocorr_k,
        'autocorr_theta1': res.autocorr_theta1,
        'time_per_iter': res.time_per_iteration,
    }

    save_result(dimension, algorithm_name, results, results_dir)

    print(f"  [OK] d={dimension:4d} {algorithm_name:10s}: "
          f"autocorr_k={res.autocorr_k:6.2f}, autocorr_theta1={res.autocorr_theta1:6.2f}, "
          f"time={res.time_per_iteration:6.2f}ms [SAVED]")

    return (dimension, algorithm_name, results)


def run_all_experiments_parallel(dimensions, n_obs=20, n_iter=2000, burn_in=200,
                                n_jobs=-1, results_dir='results'):
    algorithms = ['Alg1', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    existing, missing = check_existing_results(dimensions, algorithms, results_dir)

    print("CHECKING CACHED RESULTS")
    print(f"Total jobs: {len(dimensions)} dimensions x {len(algorithms)} algorithms = {len(dimensions) * len(algorithms)} jobs")
    print(f"Found {len(existing)} cached results")
    print(f"Missing {len(missing)} results")

    if existing:
        print(f"\nLoaded results:")
        for dim, algo, _, _ in sorted(existing)[:10]:
            print(f"  [OK] d={dim:4d} {algo:10s}")
        if len(existing) > 10:
            print(f"  ... and {len(existing) - 10} more")

    results_by_dim = {dim: {} for dim in dimensions}
    for dim, algo_name, m_val, result in existing:
        results_by_dim[dim][algo_name] = result

    if missing:
        if n_jobs == -1:
            n_processes = cpu_count()
        else:
            n_processes = min(n_jobs, cpu_count())

        print("\nPARALLEL EXECUTION")
        print(f"Using {n_processes} CPU cores")
        print(f"Computing {len(missing)} missing jobs...")
        print()

        jobs = []
        for dim, algo_name, m_val in missing:
            jobs.append((dim, algo_name, m_val, n_obs, n_iter, burn_in, results_dir))

        print("Running experiments in parallel...")
        print()
        with Pool(processes=n_processes) as pool:
            raw_results = pool.map(run_single_algorithm, jobs)

        print()
        print("All parallel jobs completed!")

        for dim, algo_name, results in raw_results:
            results_by_dim[dim][algo_name] = results
    else:
        print("\nAll results already cached! No computation needed.")

    all_results = [results_by_dim[dim] for dim in dimensions]

    return all_results


def plot_results(all_results, dimensions):
    algorithms = ['Alg1', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    time_per_iter = {algo: [] for algo in algorithms}
    autocorr_k = {algo: [] for algo in algorithms}
    autocorr_theta1 = {algo: [] for algo in algorithms}

    for results in all_results:
        for algo in algorithms:
            time_per_iter[algo].append(results[algo]['time_per_iter'])
            autocorr_k[algo].append(results[algo]['autocorr_k'])
            autocorr_theta1[algo].append(results[algo]['autocorr_theta1'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    ax = axes[1]
    for algo in algorithms:
        ax.plot(dimensions, autocorr_k[algo], marker='s',
                label=algo, linewidth=2, markersize=6)
    ax.set_xlabel('Data Dimension (d)', fontsize=12)
    ax.set_ylabel('Autocorrelation Time for k', fontsize=12)
    ax.set_title('Autocorrelation Time vs Dimension\n(Number of components k)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax = axes[2]
    for algo in algorithms:
        ax.plot(dimensions, autocorr_theta1[algo], marker='^',
                label=algo, linewidth=2, markersize=6)
    ax.set_xlabel('Data Dimension (d)', fontsize=12)
    ax.set_ylabel('Autocorrelation Time for theta_1', fontsize=12)
    ax.set_title('Autocorrelation Time vs Dimension\n(First parameter θ₁)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()

    output_path = Path('results') / 'scaling_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    plt.close()


def print_summary_table(all_results, dimensions):
    algorithms = ['Alg1', 'Alg8_m1', 'Alg8_m2', 'Alg8_m30']

    print("\nTime per Iteration (ms)")
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"d={dim:<10}", end="")
    print()
    print()

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for results in all_results:
            val = results[algo]['time_per_iter']
            print(f"{val:<10.2f}", end="")
        print()

    print("\nAutocorrelation Time for k")
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"d={dim:<10}", end="")
    print()
    print()

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for results in all_results:
            val = results[algo]['autocorr_k']
            print(f"{val:<10.2f}", end="")
        print()

    print("\nAutocorrelation Time for theta_1")
    print(f"{'Algorithm':<15}", end="")
    for dim in dimensions:
        print(f"d={dim:<10}", end="")
    print()
    print()

    for algo in algorithms:
        print(f"{algo:<15}", end="")
        for results in all_results:
            val = results[algo]['autocorr_theta1']
            print(f"{val:<10.2f}", end="")
        print()


def main():
    dimensions = [1, 2, 4, 8, 16, 32, 64]
    n_iter = 200000
    burn_in = 20000

    print(f"Dimensions to test: {dimensions}")
    print()

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

    print_summary_table(all_results, dimensions)

    # Plot results
    plot_results(all_results, dimensions)

if __name__ == "__main__":
    main()
