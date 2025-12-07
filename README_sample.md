# MCMC Sampling for Dirichlet Process Mixture Models

Implementation of the 8 MCMC algorithms from:
**Neal, R. M. (1998). "Markov Chain Sampling Methods for Dirichlet Process Mixture Models"**

## Project Structure

```
.
├── src/sample/              # Main package
│   ├── base.py             # Base classes (DirichletProcessMixture, MCMCResults)
│   ├── base_algorithm.py   # Base algorithm class with common methods
│   ├── algorithm1.py       # Algorithm 1: Basic Gibbs Sampling
│   ├── algorithm2.py       # Algorithm 2: Auxiliary Gibbs (m=30)
│   ├── algorithm3.py       # Algorithm 3: "No gaps" algorithm
│   ├── algorithm4.py       # Algorithm 4: "No gaps" with M-H
│   ├── algorithm5.py       # Algorithm 5: M-H with R proposals
│   ├── algorithm6.py       # Algorithm 6: M-H without φ update
│   ├── algorithm7.py       # Algorithm 7: Modified M-H & Partial Gibbs
│   ├── algorithm8.py       # Algorithm 8: Auxiliary parameter method
│   ├── runner.py           # MCMCRunner class to orchestrate algorithms
│   ├── visualization.py    # Visualization utilities
│   └── __init__.py         # Package initialization
├── results/                # Directory for saved results
├── run_paper_example.py    # Reproduce Table 1 from the paper
├── run_dimension_analysis.py  # Analyze scaling with dimension
├── dirichlet_mixture_mcmc.py  # Original monolithic implementation
└── README_sample.md        # This file
```

## Installation

Required packages:
```bash
pip install numpy scipy matplotlib
```

## Usage

### 1. Reproduce Paper Results (Table 1)

To reproduce the results from Table 1 in Neal (1998):

```bash
python run_paper_example.py
```

This will:
- Run all 8 algorithms on the example data from the paper
- Print comparison with expected values from Table 1
- Save results to `results/paper_example_results.pkl`
- Generate comparison plots in `results/comparison_table1.png`

### 2. Dimension Analysis

To analyze how computational cost scales with data dimension:

```bash
python run_dimension_analysis.py
```

This will:
- Test algorithms on datasets of varying sizes (10, 20, 50, 100, 200, 500)
- Compute time needed to obtain 2 independent samples
- Generate plot showing computational cost vs dimension
- Save figure to `results/iid_time_vs_dimension.png`

### 3. Custom Usage

You can use the package programmatically:

```python
import numpy as np
from src.sample import DirichletProcessMixture
from src.sample.runner import MCMCRunner

# Create data
data = np.array([-1.48, -1.40, -1.16, -1.08, -1.02,
                 0.14, 0.51, 0.53, 0.78])

# Create runner
runner = MCMCRunner(
    data=data,
    alpha=1.0,      # Concentration parameter
    sigma=0.1,      # Likelihood std dev
    mu0=0.0,        # Prior mean
    sigma0=1.0,     # Prior std dev
    results_dir='results'
)

# Run specific algorithm
results = runner.run_algorithm('Alg8_m2', n_iter=1000, burn_in=100)

# Or run all algorithms
all_results = runner.run_all(n_iter=1000, burn_in=100)

# Print summary
runner.print_summary()

# Save results
runner.save_results('my_results.pkl')

# Compute time for 2 i.i.d. samples
time_2_iid = runner.compute_iid_time('Alg8_m2', autocorr_variable='k')
print(f"Time for 2 i.i.d. samples: {time_2_iid:.2f} ms")
```

## Algorithms

The 8 algorithms implemented are:

1. **Algorithm 1**: Basic Gibbs sampling
2. **Algorithm 2**: Auxiliary Gibbs with m=30 (limit as m→∞)
3. **Algorithm 3**: "No gaps" algorithm
4. **Algorithm 4**: "No gaps" with Metropolis-Hastings
5. **Algorithm 5**: Metropolis-Hastings with R=4 proposals
6. **Algorithm 6**: M-H without φ update
7. **Algorithm 7**: Modified M-H & Partial Gibbs
8. **Algorithm 8**: Auxiliary parameter method (with m=1, 2, or 30)

## Results Interpretation

### MCMCResults Object

Each algorithm returns an `MCMCResults` object with:
- `c`: Component indicators (n_samples × n_data)
- `phi`: Final component parameters
- `theta`: Individual parameters (n_samples × n_data)
- `time_per_iteration`: Average time per iteration in milliseconds
- `autocorr_k`: Lag-1 autocorrelation of number of components
- `autocorr_theta1`: Lag-1 autocorrelation of first parameter

### Computing Independent Sample Time

The time to obtain 2 independent samples is computed as:

```
τ = 1 / (1 - ρ)  # Autocorrelation time
time_2_iid = 2 × τ × time_per_iteration
```

where ρ is the lag-1 autocorrelation.

## Visualization

The package includes several visualization functions:

### 1. Comparison with Table 1
```python
from src.sample.visualization import plot_comparison_table1

plot_comparison_table1(results, save_fig=True, results_dir='results')
```

### 2. Dimension Analysis
```python
from src.sample.visualization import plot_autocorr_time_vs_dimension

plot_autocorr_time_vs_dimension(
    dimensions=[10, 20, 50, 100],
    algorithms=['Alg3', 'Alg7', 'Alg8_m2'],
    n_iter=1000,
    burn_in=100
)
```

### 3. Trace Plots
```python
from src.sample.visualization import plot_trace_plots

plot_trace_plots(results, 'Alg8_m2', save_fig=True)
```

## Expected Results (Table 1 from Paper)

| Algorithm | Time/iter (ms) | Autocorr k | Autocorr θ₁ |
|-----------|----------------|------------|-------------|
| Alg4      | 7.6           | 13.7       | 8.5         |
| Alg5      | 8.6           | 8.1        | 10.2        |
| Alg6      | 8.3           | 19.4       | 64.1        |
| Alg7      | 8.0           | 6.9        | 5.3         |
| Alg8_m1   | 7.9           | 5.2        | 5.6         |
| Alg8_m2   | 8.8           | 3.7        | 4.7         |
| Alg8_m30  | 38.0          | 2.0        | 2.8         |

Note: Your results may differ slightly due to randomness and hardware differences.

## Reference

Neal, R. M. (1998). Markov Chain Sampling Methods for Dirichlet Process Mixture Models.
*Journal of Computational and Graphical Statistics*, 9(2), 249-265.
