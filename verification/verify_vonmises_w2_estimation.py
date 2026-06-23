"""Verification of von Mises parameter estimation using 2-Wasserstein distance.

This script compares parameter estimation via 2-Wasserstein distance minimization
using two different approximation methods for the target continuous distribution:
1. Random sampling approximation (scipy.stats.vonmises.rvs)
2. Hybrid quantile sampling approximation (deterministic representatives)

It also compares the performance and execution time with MLE.
"""

import time
from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from src.distributions import vonmises
from src.method import circular_wasserstein_from_samples


def estimate_by_random_sampling_w2(
    given_data: np.ndarray,
) -> Tuple[float, float]:
    """Estimate von Mises parameters using W2 distance with random sampling.

    Approximates the model distribution by generating random samples at each
    optimization iteration.

    Args:
        given_data (np.ndarray): Observed circular samples in [-pi, pi].

    Returns:
        Tuple[float, float]: Estimated [mu, kappa].
    """
    given_data_norm = given_data / (2 * np.pi)
    n_samples = len(given_data)

    def cost_func(params: np.ndarray) -> float:
        mu, kappa = params
        # Generate random samples from the candidate model distribution
        model_sample = stats.vonmises(loc=mu, kappa=kappa).rvs(n_samples) / (2 * np.pi)
        return circular_wasserstein_from_samples(given_data_norm, model_sample, p=2)

    # Use grid search (brute) only for quick verification
    res = optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0.01, 10.0)),
        full_output=True,
        finish=None,
        Ns=10,
    )
    return float(res[0][0]), float(res[0][1])


def estimate_by_quantile_sampling_w2(
    given_data: np.ndarray,
) -> Tuple[float, float]:
    """Estimate von Mises parameters using W2 distance with quantile sampling.

    Approximates the model distribution deterministically using hybrid
    quantile sampling, which is faster and numerically stable.

    Args:
        given_data (np.ndarray): Observed circular samples in [-pi, pi].

    Returns:
        Tuple[float, float]: Estimated [mu, kappa].
    """
    given_data_norm = given_data / (2 * np.pi)
    n_samples = len(given_data)

    def cost_func(params: np.ndarray) -> float:
        mu, kappa = params
        # Deterministically select representative points using quantile sampling
        model_sample = np.remainder(
            vonmises.quantile_sampling(mu, kappa, n_samples), 2 * np.pi
        ) / (2 * np.pi)
        return circular_wasserstein_from_samples(given_data_norm, model_sample, p=2)

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

    n_samples = 100
    true_mu = 0.3
    true_kappa = 2.0

    print(
        f"--- von Mises W2 Estimation Verification ---"
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

    # 2. W2 Estimation via Random Sampling
    start_time = time.perf_counter()
    est_mu_rand, est_kappa_rand = estimate_by_random_sampling_w2(observed_data)
    rand_time = time.perf_counter() - start_time
    print(
        f"W2 (Random Sampling):    mu = {est_mu_rand:.5f}, "
        f"kappa = {est_kappa_rand:.5f} (Time: {rand_time:.4f}s)"
    )

    # 3. W2 Estimation via Quantile Sampling
    start_time = time.perf_counter()
    est_mu_quant, est_kappa_quant = estimate_by_quantile_sampling_w2(observed_data)
    quant_time = time.perf_counter() - start_time
    print(
        f"W2 (Quantile Sampling):  mu = {est_mu_quant:.5f}, "
        f"kappa = {est_kappa_quant:.5f} (Time: {quant_time:.4f}s)"
    )


if __name__ == "__main__":
    main()
