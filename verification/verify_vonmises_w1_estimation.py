"""Verification of von Mises parameter estimation using 1-Wasserstein distance.

This script validates parameter estimation via 1-Wasserstein distance minimization
using the cumulative sum histogram of equal-division bins (Method 2).

It compares the performance and execution time with Maximum Likelihood Estimation (MLE).
"""

import time
from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from src.distributions import vonmises
from src.method import circular_w1_from_cumsums


def estimate_by_cumsum_w1(given_data: np.ndarray) -> Tuple[float, float]:
    """Estimate von Mises parameters using W1 distance from cumulative sum histograms.

    Args:
        given_data (np.ndarray): Observed circular samples in [-pi, pi].

    Returns:
        Tuple[float, float]: Estimated [mu, kappa].
    """
    bin_num = len(given_data)
    # Compute the cumulative distribution histogram of the observed data
    data_cumsum_hist = vonmises.cumsum_hist_data(given_data, bin_num)

    def cost_func(params: np.ndarray) -> float:
        mu, kappa = params
        dist_cumsum_hist = vonmises.cumsum_hist(mu, kappa, bin_num)
        # Compare histograms using fast W1 calculation
        return circular_w1_from_cumsums(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    # Use grid search (brute) followed by Powell optimization for refinement
    res = optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0.01, 10.0)),
        full_output=True,
        finish=optimize.fmin_powell,
        Ns=20,
    )
    return float(res[0][0]), float(res[0][1])


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)

    n_samples = 10000
    true_mu = 0.3
    true_kappa = 2.0

    print(
        f"--- von Mises W1 Estimation Verification ---"
        f"\nSample Size (N) = {n_samples}"
        f"\nTrue Parameters: mu = {true_mu}, kappa = {true_kappa}\n"
    )

    # Generate synthetic observations
    observed_data = stats.vonmises(loc=true_mu, kappa=true_kappa).rvs(n_samples)

    # 1. Maximum Likelihood Estimation (MLE)
    start_time = time.perf_counter()
    t_statistics = vonmises.T(observed_data)
    mle_mu, mle_kappa = vonmises.MLE(t_statistics, n_samples)
    mle_time = time.perf_counter() - start_time
    print(
        f"MLE Result:              mu = {mle_mu:.5f}, kappa = {mle_kappa:.5f} "
        f"(Time: {mle_time:.4f}s)"
    )

    # 2. W1 Estimation via Cumulative Sum Histograms (Method 2)
    start_time = time.perf_counter()
    est_mu_w1, est_kappa_w1 = estimate_by_cumsum_w1(observed_data)
    w1_time = time.perf_counter() - start_time
    print(
        f"W1 (Method 2):           mu = {est_mu_w1:.5f}, kappa = {est_kappa_w1:.5f} "
        f"(Time: {w1_time:.4f}s)"
    )


if __name__ == "__main__":
    main()
