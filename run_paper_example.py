"""
Validate Algorithm 8 against Neal (1998) Table 1

This script reproduces the experimental setup for Algorithm 8 from Section 8
of the paper, using proper initialization with Algorithm 5 as specified.

Reference values from Table 1 (Neal 1998, page 15):
- Time is in milliseconds per iteration
- Autocorrelation time for k (number of clusters)
- Autocorrelation time for theta_1
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample.algorithm5 import Algorithm5
from src.sample.algorithm8 import Algorithm8
from src.sample.base import DirichletProcessMixture

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# EXPERIMENTAL SETUP FROM NEAL (1998), SECTION 8
# ============================================================================

# Data from the paper (page 14-15)
data = np.array([-1.48, -1.40, -1.16, -1.08, -1.02,
                 0.14, 0.51, 0.53, 0.78])

# Model parameters from the paper
# F(θ) = N(θ, σ²) with σ² = 0.01
# G₀ = N(0, 1)
# α = 1
alpha = 1.0
sigma = 0.1  # σ² = 0.01
mu0 = 0.0
sigma0 = 1.0

print("=" * 80)
print("ALGORITHM 8 VALIDATION - Neal (1998) Table 1")
print("=" * 80)
print()
print(f"Model parameters:")
print(f"  - n = {len(data)}")
print(f"  - σ² = {sigma**2}")
print(f"  - μ₀ = {mu0}")
print(f"  - σ₀² = {sigma0**2}")
print(f"  - α = {alpha}")
print()
print(f"Data: {data}")
print()

# Create model
model = DirichletProcessMixture(data=data, alpha=alpha, sigma=sigma,
                               mu0=mu0, sigma0=sigma0)

# Reference values from Table 1 (Neal 1998, page 15)
# Format: (time_ms, autocorr_k, autocorr_theta1)
REFERENCE_RESULTS = {
    'Alg8_m1': (7.9, 5.2, 5.6),
    'Alg8_m2': (8.8, 3.7, 4.7),
    'Alg8_m30': (38.0, 2.0, 2.8),
}

# ============================================================================
# INITIALIZATION WITH ALGORITHM 5
# ============================================================================

print("-" * 80)
print("Step 1: Initialization with Algorithm 5 (100 iterations)")
print("-" * 80)

# Run Algorithm 5 for 100 iterations to get initial state
# The paper uses this to initialize Algorithm 8
algo5 = Algorithm5(model=model)
results5 = algo5.run(n_iter=100, burn_in=0, R=4)

# Extract final state from Algorithm 5
initial_c = results5.c[-1].copy()
initial_theta = results5.theta[-1].copy()

# Reconstruct phi from the final state
unique_c = np.unique(initial_c)
initial_phi = []
for uc in unique_c:
    mask = (initial_c == uc)
    # Take the mean of theta values in this cluster as phi
    initial_phi.append(np.mean(initial_theta[mask]))

# Create initial state tuple
initial_state = (initial_c, initial_phi, initial_theta)

print(f"✓ Initialization complete")
print(f"  - Initial number of clusters: {len(initial_phi)}")
print()

# ============================================================================
# RUN ALGORITHM 8 WITH DIFFERENT VALUES OF m
# ============================================================================

# Number of iterations for main run
# Paper uses 20,000 iterations after initialization
n_iter = 20000
burn_in = 0  # Already initialized with Algorithm 5

print("-" * 80)
print(f"Step 2: Run Algorithm 8 ({n_iter} iterations)")
print("-" * 80)
print()

results = {}

for m_value in [1, 2, 30]:
    print(f"Running Algorithm 8 with m={m_value}...")

    # Create fresh algorithm instance
    algo8 = Algorithm8(model=model)

    # Create a fresh copy of initial state for each run
    initial_state_copy = (initial_c.copy(), initial_phi.copy(), initial_theta.copy())

    # Run Algorithm 8 with initialization from Algorithm 5
    result = algo8.run(n_iter=n_iter, burn_in=burn_in, m=m_value,
                      initial_state=initial_state_copy)

    # Store results
    alg_name = f'Alg8_m{m_value}'
    results[alg_name] = result

    print(f"✓ Completed")
    print(f"  - Final number of clusters: {len(np.unique(result.c[-1]))}")
    print(f"  - Time per iteration: {result.time_per_iteration:.2f} ms")
    print(f"  - Autocorr time (k): {result.autocorr_k:.3f}")
    print(f"  - Autocorr time (θ₁): {result.autocorr_theta1:.3f}")
    print()

# ============================================================================
# COMPARISON WITH PAPER
# ============================================================================

print("=" * 80)
print("COMPARISON WITH NEAL (1998) TABLE 1")
print("=" * 80)
print()

print(f"{'Algorithm':<15} {'Metric':<20} {'Paper':<12} {'Ours':<12}")
print("-" * 80)

for alg_name in ['Alg8_m1', 'Alg8_m2', 'Alg8_m30']:
    exp_time, exp_k, exp_theta = REFERENCE_RESULTS[alg_name]
    res = results[alg_name]

    # Time comparison
    print(f"{alg_name:<15} {'Time (ms/iter)':<20} {exp_time:<12.1f} "
          f"{res.time_per_iteration:<12.2f}")

    # Autocorr k comparison
    print(f"{'':<15} {'Autocorr(k)':<20} {exp_k:<12.1f} "
          f"{res.autocorr_k:<12.3f}")

    # Autocorr theta1 comparison
    print(f"{'':<15} {'Autocorr(θ₁)':<20} {exp_theta:<12.1f} "
          f"{res.autocorr_theta1:<12.3f}")
    print()

# Check autocorrelation trends
trend_k = (results['Alg8_m30'].autocorr_k < results['Alg8_m2'].autocorr_k <
           results['Alg8_m1'].autocorr_k)
trend_theta = (results['Alg8_m30'].autocorr_theta1 < results['Alg8_m2'].autocorr_theta1 <
               results['Alg8_m1'].autocorr_theta1)

if trend_k:
    print("✓ Autocorr(k) trend matches paper: m=30 < m=2 < m=1")
else:
    print("✗ Autocorr(k) trend differs from paper")
    print(f"  Observed: m=30={results['Alg8_m30'].autocorr_k:.2f}, "
          f"m=2={results['Alg8_m2'].autocorr_k:.2f}, "
          f"m=1={results['Alg8_m1'].autocorr_k:.2f}")

if trend_theta:
    print("✓ Autocorr(θ₁) trend matches paper: m=30 < m=2 < m=1")
else:
    print("✗ Autocorr(θ₁) trend differs from paper")
    print(f"  Observed: m=30={results['Alg8_m30'].autocorr_theta1:.2f}, "
          f"m=2={results['Alg8_m2'].autocorr_theta1:.2f}, "
          f"m=1={results['Alg8_m1'].autocorr_theta1:.2f}")