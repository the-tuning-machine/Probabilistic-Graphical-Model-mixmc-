# Markov Chain Sampling Methods for Dirichlet Process Mixture Models

Implementation of the 8 MCMC algorithms presented in:

**Neal, R. M. (1998). "Markov Chain Sampling Methods for Dirichlet Process Mixture Models"**
Technical Report No. 9815, Department of Statistics, University of Toronto

## Overview

This project implements and compares 8 different Markov Chain Monte Carlo (MCMC) algorithms for sampling from the posterior distribution of Dirichlet Process Mixture Models.

## Model Specification

The Dirichlet Process Mixture Model is specified as:

```
y_i | θ_i ~ F(θ_i)
θ_i | G ~ G
G ~ DP(G_0, α)
```

For the example in the paper:
- **F(θ) = N(θ, σ²)** with σ² = 0.01 (σ = 0.1)
- **G₀ = N(0, 1)** (standard normal prior)
- **α = 1.0** (concentration parameter)

## Dataset

The dataset used (from page 14-15 of the paper):

```python
y = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
```

This dataset has n=9 observations and is expected to have around 2-3 mixture components.

## Algorithms Implemented

### 1. Algorithm 1: Basic Gibbs Sampling
Standard Gibbs sampler that updates component indicators c_i and parameters φ_c.

### 2. Algorithm 2: Gibbs with Auxiliary Parameters (m→∞)
Limit case of Algorithm 8 as m approaches infinity (approximated with m=30).

### 3. Algorithm 3: "No Gaps" Algorithm
Maintains components numbered 1,...,k without gaps, simplifying bookkeeping.

### 4. Algorithm 4: "No Gaps" Variant
Similar to Algorithm 3 with careful relabeling strategy.

### 5. Algorithm 5: Metropolis-Hastings (R=4)
Uses Metropolis-Hastings updates with R proposals per observation.
- Proposes new c_i from conditional prior
- Accepts/rejects based on likelihood ratio

### 6. Algorithm 6: M-H without φ Update (R=4)
Similar to Algorithm 5 but without updating the φ_c parameters.

### 7. Algorithm 7: Modified M-H & Partial Gibbs
Combines Metropolis-Hastings updates for c_i with Gibbs sampling for φ_c.

### 8. Algorithm 8: Auxiliary Parameter Method (m=1, 2, 30)
Introduces m auxiliary parameters to improve mixing:
- For each observation, draws m auxiliary parameters from G₀
- Samples c_i from augmented conditional distribution
- Varying m provides different exploration-exploitation tradeoffs

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run all algorithms and compare results:

```bash
python dirichlet_mixture_mcmc.py
```

This will:
1. Run all 8 algorithm variants
2. Compute performance metrics (time per iteration, autocorrelations)
3. Compare results with Table 1 from the paper

### Generate visualizations:

```bash
python visualize_results.py
```

This creates:
- `mcmc_comparison_components.png` - Trace plots of number of components
- `mcmc_comparison_theta1.png` - Trace plots of θ₁ parameter
- `mcmc_comparison_stats.png` - Bar charts of performance metrics
- `mcmc_comparison_posterior.png` - Posterior distributions

## Results Comparison with Table 1

The paper presents the following performance metrics (Table 1, page 15):

| Algorithm | Time/iter (μs) | Autocorr k | Autocorr θ₁ |
|-----------|---------------|------------|-------------|
| Alg. 4 ("no gaps") | 7.6 | 13.7 | 8.5 |
| Alg. 5 (M-H, R=4) | 8.6 | 8.1 | 10.2 |
| Alg. 6 (M-H, R=4, no φ) | 8.3 | 19.4 | 64.1 |
| Alg. 7 (mod M-H & Gibbs) | 8.0 | 6.9 | 5.3 |
| Alg. 8 (aux, m=1) | 7.9 | 5.2 | 5.6 |
| Alg. 8 (aux, m=2) | 8.8 | 3.7 | 4.7 |
| Alg. 8 (aux, m=30) | 38.0 | 2.0 | 2.8 |

### Key Findings:

1. **Computational Cost**:
   - Algorithms 1-7 have similar computational costs
   - Algorithm 8 with m=30 is significantly slower (≈5x)

2. **Mixing Efficiency** (measured by autocorrelation):
   - Algorithm 8 with larger m has better mixing (lower autocorrelation)
   - Algorithm 6 (without φ update) has poor mixing for θ
   - Trade-off between computational cost and mixing quality

3. **Recommendations**:
   - Algorithm 8 with m=2 or m=3 provides good balance
   - Algorithm 7 is efficient for conjugate priors
   - Avoid Algorithm 6 (no φ update) for general use

## Files

- `dirichlet_mixture_mcmc.py` - Main implementation of all 8 algorithms
- `visualize_results.py` - Visualization script
- `requirements.txt` - Python dependencies
- `mixmc.pdf` - Original paper by Neal (1998)
- `README.md` - This file

## Mathematical Details

### Dirichlet Process Prior

The Dirichlet Process DP(G₀, α) is a distribution over distributions characterized by:
- **Base distribution G₀**: Center of the DP
- **Concentration α**: Controls variability around G₀

### Conjugate Update

For the normal-normal conjugate case:

Posterior for θ | y:
```
1/σ²_post = 1/σ² + 1/σ²₀
μ_post = (y/σ² + μ₀/σ²₀) / (1/σ²_post)
θ | y ~ N(μ_post, σ²_post)
```

### Component Indicator Update

Full conditional for c_i:

```
P(c_i = c | c_{-i}, y_i, φ) ∝ {
    n_{-i,c} · F(y_i | φ_c)  if c exists
    α · ∫ F(y_i | φ) dG₀(φ)   if c is new
}
```

where n_{-i,c} is the number of observations (excluding i) in component c.

## References

1. Neal, R. M. (1998). "Markov Chain Sampling Methods for Dirichlet Process Mixture Models", Technical Report No. 9815, Department of Statistics, University of Toronto.

2. Escobar, M. D., & West, M. (1995). "Bayesian density estimation and inference using mixtures", Journal of the American Statistical Association, 90(430), 577-588.

3. MacEachern, S. N., & Müller, P. (1998). "Estimating mixture of Dirichlet process models", Journal of Computational and Graphical Statistics, 7(2), 223-238.

## License

This implementation is for educational purposes based on the publicly available paper by Neal (1998).

## Author

Implementation based on Neal (1998) paper
Course: Probabilistic Graphical Models, ENS
