"""Verification of Wrapped Cauchy parameter estimation.

This script compares parameter estimation via 1-Wasserstein distance minimization
(using equal-division cumulative sum histograms) against MLE
for the Wrapped Cauchy distribution.
"""

import time
from functools import partial
from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from src.distributions import wrappedcauchy
from src.method import circular_w1_from_cumsums


def estimate_by_cumsum_w1(given_data: np.ndarray) -> Tuple[float, float]:
    """Estimate Wrapped Cauchy parameters using W1 distance from cumulative sum
    histograms.

    Args:
        given_data (np.ndarray): Observed circular samples in [0, 2*pi].

    Returns:
        Tuple[float, float]: Estimated [mu, rho].
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrappedcauchy.cumsum_hist_data(given_data, bin_num)

    def cost_func(params: np.ndarray) -> float:
        mu, rho = params
        dist_cumsum_hist = wrappedcauchy.cumsum_hist(mu, rho, bin_num)
        return circular_w1_from_cumsums(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    bounds = ((0.0, 2.0 * np.pi), (0.01, 0.99))
    finish_func = partial(optimize.minimize, method="powell", bounds=bounds)

    res = optimize.brute(
        cost_func,
        bounds,
        full_output=True,
        finish=finish_func,
        Ns=20,
    )
    return float(res[0][0]), float(res[0][1])


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)

    n_samples = 10000
    true_mu = np.pi / 2
    true_rho = 0.1

    print(
        f"--- Wrapped Cauchy Estimation Verification ---"
        f"\nSample Size (N) = {n_samples}"
        f"\nTrue Parameters: mu = {true_mu:.5f}, rho = {true_rho:.5f}\n"
    )

    # Generate synthetic observations and wrap them into [0, 2*pi]
    observed_data = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(n_samples)
    observed_data = np.remainder(observed_data, 2.0 * np.pi)

    # 1. Maximum Likelihood Estimation (MLE) using Okamura's method
    start_time = time.perf_counter()
    mle_result = wrappedcauchy.MLE_OKAMURA(observed_data, n_samples, iter_num=100)
    mle_time = time.perf_counter() - start_time
    mle_mu = float(mle_result[0])
    mle_rho = float(mle_result[1])
    print(
        f"MLE (Okamura):           mu = {mle_mu:.5f}, rho = {mle_rho:.5f} "
        f"(Time: {mle_time:.4f}s)"
    )

    # 2. W1 Estimation via Cumulative Sum Histograms (Method 2)
    start_time = time.perf_counter()
    est_mu_w1, est_rho_w1 = estimate_by_cumsum_w1(observed_data)
    w1_time = time.perf_counter() - start_time
    print(
        f"W1 (Method 2):           mu = {est_mu_w1:.5f}, rho = {est_rho_w1:.5f} "
        f"(Time: {w1_time:.4f}s)"
    )


if __name__ == "__main__":
    main()
