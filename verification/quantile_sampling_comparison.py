import time
import numpy as np
import scipy.stats as stats
import numba
from scipy.special import ive

from src.distributions import vonmises
from src.utils.circular_utils import to_2pi_range

# Use a coarse grid of size 16384 for the hybrid methods
M_GRID = 16384

@numba.njit(fastmath=True, cache=True)
def _find_lefts_with_indices_numba(
    x: numba.float64[:], z: numba.float64[:], step_grid: float
):
    n_x = len(x)
    n_z = len(z)
    lefts = np.empty(n_x, dtype=np.float64)
    lefts_idx = np.empty(n_x, dtype=np.intp)
    i = 0
    for j in range(n_x):
        xi = x[j]
        while i < n_z and z[i] < xi:
            i += 1
        if i == 0:
            lefts_idx[j] = 0
            lefts[j] = -np.pi
        else:
            lefts_idx[j] = i - 1
            lefts[j] = -np.pi + (i - 1) * step_grid
    return lefts, lefts_idx

def hybrid_quantile_sampling(
    mu: float, kappa: float, sample_num: int, steps: int = 1
) -> np.ndarray:
    x, step = np.linspace(0, 1, sample_num, endpoint=False, retstep=True)
    x = x + step / 2

    # 1. Evaluate CDF on a coarse grid at mu=0
    y0, step_grid = np.linspace(-np.pi, np.pi, M_GRID, retstep=True)
    n_terms = 150
    n_arr = np.arange(1, n_terms + 1)
    r_n = ive(n_arr, kappa) / ive(0, kappa)
    
    # base is always -0.5, so F(y0) - base = F(y0) + 0.5
    z = vonmises._vonmises_cdf_series_numba_mu0(y0, r_n) + 0.5
    
    # 2. Linear search to get left indices
    lefts, lefts_idx = _find_lefts_with_indices_numba(x, z, step_grid)
    
    # Linear interpolation to construct initial guess theta_0
    theta = np.empty(sample_num, dtype=np.float64)
    for j in range(sample_num):
        idx = lefts_idx[j]
        if idx >= M_GRID - 1:
            theta[j] = np.pi
        else:
            z_l = z[idx]
            z_r = z[idx + 1]
            denom = z_r - z_l
            if denom < 1e-15:
                theta[j] = lefts[j]
            else:
                theta[j] = lefts[j] + (x[j] - z_l) / denom * step_grid
                
    # 3. Newton-Raphson iterations
    i0_kappa = ive(0, kappa)
    for _ in range(steps):
        # Evaluate CDF using the Clenshaw mu=0 JIT function
        cdf_vals = vonmises._vonmises_cdf_series_numba_mu0(theta, r_n) + 0.5
        # Evaluate PDF
        pdf_vals = np.exp(kappa * (np.cos(theta) - 1.0)) / (2.0 * np.pi * i0_kappa)
        pdf_vals = np.clip(pdf_vals, 1e-15, None)  # avoid division by zero
        
        # Newton update step
        theta = theta - (cdf_vals - x) / pdf_vals
        
    # Map to [0, 2pi] and perform topology correction
    samples = to_2pi_range(theta + mu)
    if sample_num > 1:
        diffs = np.diff(samples)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0:
            shift = min_idx + 1
        else:
            shift = 0
        samples = np.roll(samples, -shift)
        
    return samples

def run_comparison():
    # Setup test parameters
    mu = 0.3
    kappa = 2.0
    scales = [1000, 10000, 100000]
    
    # Compile Numba helper functions on first call (warmup)
    print("Warming up Numba JIT compilers...")
    _ = hybrid_quantile_sampling(mu, kappa, 10, steps=1)
    _ = hybrid_quantile_sampling(mu, kappa, 10, steps=2)
    _ = vonmises.fast_quantile_sampling(mu, kappa, 10)
    
    print("\nRunning benchmarks...")
    print(f"{'Method':<30} | {'N':<6} | {'Time (s)':<10} | {'Max Abs Error':<18}")
    print("-" * 75)
    
    for N in scales:
        # 1. SciPy
        t0 = time.perf_counter()
        samples_scipy = vonmises.quantile_sampling(mu, kappa, N)
        t_scipy = time.perf_counter() - t0
        
        # 2. Grid (Original fast_quantile_sampling, 1M points)
        t0 = time.perf_counter()
        samples_grid = vonmises.fast_quantile_sampling(mu, kappa, N)
        t_grid = time.perf_counter() - t0
        
        # Calculate error of grid method against scipy
        err_grid = np.max(np.minimum(np.abs(samples_grid - samples_scipy), 2 * np.pi - np.abs(samples_grid - samples_scipy)))
        print(f"{'quantile_sampling (SciPy)':<30} | {N:<6} | {t_scipy:<10.4f} | {'0.00e+00 (Reference)':<18}")
        print(f"{'fast_quantile_sampling (Grid)':<30} | {N:<6} | {t_grid:<10.4f} | {err_grid:<18.2e}")
        
        # 3. 1-step Newton
        t0 = time.perf_counter()
        samples_newton1 = hybrid_quantile_sampling(mu, kappa, N, steps=1)
        t_newton1 = time.perf_counter() - t0
        
        err_newton1 = np.max(np.minimum(np.abs(samples_newton1 - samples_scipy), 2 * np.pi - np.abs(samples_newton1 - samples_scipy)))
        print(f"{'1-step Newton (Coarse Grid)':<30} | {N:<6} | {t_newton1:<10.4f} | {err_newton1:<18.2e}")
        
        # 4. 2-step Newton
        t0 = time.perf_counter()
        samples_newton2 = hybrid_quantile_sampling(mu, kappa, N, steps=2)
        t_newton2 = time.perf_counter() - t0
        
        err_newton2 = np.max(np.minimum(np.abs(samples_newton2 - samples_scipy), 2 * np.pi - np.abs(samples_newton2 - samples_scipy)))
        print(f"{'2-step Newton (Coarse Grid)':<30} | {N:<6} | {t_newton2:<10.4f} | {err_newton2:<18.2e}")
        print("-" * 75)

if __name__ == "__main__":
    run_comparison()
