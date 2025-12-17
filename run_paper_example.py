import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.sample.algorithm5 import Algorithm5
from src.sample.algorithm8 import Algorithm8
from src.sample.base import DirichletProcessMixture

# Data from the paper
data = np.array([-1.48, -1.40, -1.16, -1.08, -1.02,
                 0.14, 0.51, 0.53, 0.78])

REFERENCE_RESULTS = {
    'Alg8_m1': (7.9, 5.2, 5.6),
    'Alg8_m2': (8.8, 3.7, 4.7),
    'Alg8_m30': (38.0, 2.0, 2.8),
}

alpha = 1.0
sigma = 0.1
mu0 = 0.0
sigma0 = 1.0

print("ALGORITHM 8 VALIDATION - Neal (1998) Table 1")
print()
print("Model parameters:")
print(f"  n = {len(data)}")
print(f"  sigma^2 = {sigma**2}")
print(f"  mu_0 = {mu0}")
print(f"  sigma_0^2 = {sigma0**2}")
print(f"  alpha = {alpha}")
print()
print(f"Data: {data}")
print()

model = DirichletProcessMixture(data=data, alpha=alpha, sigma=sigma,
                               mu0=mu0, sigma0=sigma0)

print()
print("Step 1: Initialization with Algorithm 5 (100 iterations)")
print()

algo5 = Algorithm5(model=model)
results5 = algo5.run(n_iter=100, burn_in=0, R=4)

initial_c = results5.c[-1].copy()
initial_theta = results5.theta[-1].copy()

unique_c = np.unique(initial_c)
initial_phi = []
for uc in unique_c:
    mask = (initial_c == uc)
    initial_phi.append(np.mean(initial_theta[mask]))

initial_state = (initial_c, initial_phi, initial_theta)

print("Initialization complete")
print(f"  Initial number of clusters: {len(initial_phi)}")
print()

n_iter = 20000
burn_in = 0

print()
print(f"Step 2: Run Algorithm 8 ({n_iter} iterations)")
print()

results = {}

for m_value in [1, 2, 30]:
    print(f"Running Algorithm 8 with m={m_value}...")

    algo8 = Algorithm8(model=model)
    initial_state_copy = (initial_c.copy(), initial_phi.copy(), initial_theta.copy())
    result = algo8.run(n_iter=n_iter, burn_in=burn_in, m=m_value,
                      initial_state=initial_state_copy)

    alg_name = f'Alg8_m{m_value}'
    results[alg_name] = result

    print("Completed")
    print(f"  Final number of clusters: {len(np.unique(result.c[-1]))}")
    print(f"  Time per iteration: {result.time_per_iteration:.2f} ms")
    print(f"  Autocorr time (k): {result.autocorr_k:.3f}")
    print(f"  Autocorr time (theta_1): {result.autocorr_theta1:.3f}")
    print()

print()
print("COMPARISON WITH NEAL (1998) TABLE 1")
print()

print(f"{'Algorithm':<15} {'Metric':<20} {'Paper':<12} {'Ours':<12}")
print()

for alg_name in ['Alg8_m1', 'Alg8_m2', 'Alg8_m30']:
    exp_time, exp_k, exp_theta = REFERENCE_RESULTS[alg_name]
    res = results[alg_name]

    print(f"{alg_name:<15} {'Time (ms/iter)':<20} {exp_time:<12.1f} "
          f"{res.time_per_iteration:<12.2f}")
    print(f"{'':<15} {'Autocorr(k)':<20} {exp_k:<12.1f} "
          f"{res.autocorr_k:<12.3f}")
    print(f"{'':<15} {'Autocorr(theta_1)':<20} {exp_theta:<12.1f} "
          f"{res.autocorr_theta1:<12.3f}")
    print()
trend_k = (results['Alg8_m30'].autocorr_k < results['Alg8_m2'].autocorr_k <
           results['Alg8_m1'].autocorr_k)
trend_theta = (results['Alg8_m30'].autocorr_theta1 < results['Alg8_m2'].autocorr_theta1 <
               results['Alg8_m1'].autocorr_theta1)

if trend_k:
    print("[OK] Autocorr(k) trend matches paper: m=30 < m=2 < m=1")
else:
    print("[FAILED] Autocorr(k) trend differs from paper")
    print(f"  Observed: m=30={results['Alg8_m30'].autocorr_k:.2f}, "
          f"m=2={results['Alg8_m2'].autocorr_k:.2f}, "
          f"m=1={results['Alg8_m1'].autocorr_k:.2f}")

if trend_theta:
    print("[OK] Autocorr(theta_1) trend matches paper: m=30 < m=2 < m=1")
else:
    print("[FAILED] Autocorr(theta_1) trend differs from paper")
    print(f"  Observed: m=30={results['Alg8_m30'].autocorr_theta1:.2f}, "
          f"m=2={results['Alg8_m2'].autocorr_theta1:.2f}, "
          f"m=1={results['Alg8_m1'].autocorr_theta1:.2f}")