"""Loss surface verification and visualization for Wrapped Cauchy distribution.

This script visualizes the loss landscapes of circular 1-Wasserstein, 2-Wasserstein,
and 2-Wasserstein (with quantile sampling) distances over parameters (mu, rho)
of the Wrapped Cauchy distribution.
It demonstrates behavior under low concentration (rho = 0.1) where local optimization
methods like Powell perform better than Nelder-Mead.
"""

import time
from typing import Any, Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from src.distributions import wrappedcauchy
from src.method import (
    circular_w1_from_cumsums,
    circular_wasserstein_from_samples,
)
from src.plots import brute_heatmap


def calculate_w1_loss_surface(given_data: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    """Calculate the W1 loss surface (Method 2) over a grid of parameters.

    Args:
        given_data (np.ndarray): Observed circular samples in [0, 2*pi].

    Returns:
        Tuple: Optimization result tuple.
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrappedcauchy.cumsum_hist_data(given_data, bin_num)

    def cost_func(params: np.ndarray) -> float:
        mu, rho = params
        dist_cumsum_hist = wrappedcauchy.cumsum_hist(mu, rho, bin_num)
        return circular_w1_from_cumsums(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.brute(
        cost_func,
        ((0.0, 2.0 * np.pi), (0.01, 0.99)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def calculate_w2_random_loss_surface(
    given_data: np.ndarray,
) -> Tuple[Any, Any, Any, Any]:
    """Calculate the W2 (random sampling) loss surface over a grid of parameters.

    Args:
        given_data (np.ndarray): Observed circular samples in [0, 2*pi].

    Returns:
        Tuple: Optimization result tuple.
    """
    given_data_norm = np.remainder(given_data, 2.0 * np.pi) / (2.0 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)

    def cost_func(params: np.ndarray) -> float:
        mu, rho = params
        model_sample = stats.wrapcauchy(loc=mu, c=rho).rvs(len(given_data))
        model_sample = np.remainder(model_sample, 2.0 * np.pi) / (2.0 * np.pi)
        model_sample = np.sort(model_sample)
        return circular_wasserstein_from_samples(
            given_data_norm_sorted, model_sample, p=2, sorted=True
        )

    return optimize.brute(
        cost_func,
        ((0.0, 2.0 * np.pi), (0.01, 0.99)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def calculate_w2_quantile_loss_surface(
    given_data: np.ndarray,
) -> Tuple[Any, Any, Any, Any]:
    """Calculate the W2 (quantile sampling) loss surface over a grid of parameters.

    Args:
        given_data (np.ndarray): Observed circular samples in [0, 2*pi].

    Returns:
        Tuple: Optimization result tuple.
    """
    given_data_norm = np.remainder(given_data, 2.0 * np.pi) / (2.0 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)

    def cost_func(params: np.ndarray) -> float:
        mu, rho = params
        model_sample = wrappedcauchy.quantile_sampling(mu, rho, len(given_data)) / (
            2.0 * np.pi
        )
        model_sample = np.sort(model_sample)
        return circular_wasserstein_from_samples(
            given_data_norm_sorted, model_sample, p=2, sorted=True
        )

    return optimize.brute(
        cost_func,
        ((0.0, 2.0 * np.pi), (0.01, 0.99)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def estimate_w1_powell(given_data: np.ndarray) -> optimize.OptimizeResult:
    """Run local optimization (Powell) for W1 estimation.

    Args:
        given_data (np.ndarray): Observed circular samples in [0, 2*pi].

    Returns:
        OptimizeResult: Local optimization details.
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrappedcauchy.cumsum_hist_data(given_data, bin_num)

    def cost_func(params: np.ndarray) -> float:
        mu, rho = params
        dist_cumsum_hist = wrappedcauchy.cumsum_hist(mu, rho, bin_num)
        return circular_w1_from_cumsums(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.minimize(
        cost_func,
        (0.0, 0.5),
        bounds=((0.0, 2.0 * np.pi), (0.01, 0.99)),
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)

    n_samples = 500  # Reduced for faster interactive visualization
    true_mu = 3.0 * np.pi / 2.0
    true_rho = 0.1

    print(
        f"--- Wrapped Cauchy Loss Surface Verification ---"
        f"\nSample Size (N) = {n_samples}"
        f"\nTrue Parameters: mu = {true_mu:.5f}, rho = {true_rho:.5f}\n"
    )

    # Generate synthetic observations and wrap them into [0, 2*pi]
    observed_data = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(n_samples)
    observed_data = np.remainder(observed_data, 2.0 * np.pi)

    start_time = time.perf_counter()
    mle_result = wrappedcauchy.MLE_OKAMURA(observed_data, n_samples, iter_num=100)
    mle_time = time.perf_counter() - start_time
    print(
        f"MLE Result: mu = {np.angle(mle_result):.5f}, rho = {np.abs(mle_result):.5f} "
        f"(Time: {mle_time:.4f}s)\n"
    )

    print("1. Calculating W1 loss surface (Method 2)...")
    start_time = time.perf_counter()
    w1_grid = calculate_w1_loss_surface(observed_data)
    print(f"   Done in {time.perf_counter() - start_time:.4f}s\n")

    print("2. Calculating W2 loss surface via random sampling (Method 1 - slow)...")
    start_time = time.perf_counter()
    w2_rand_grid = calculate_w2_random_loss_surface(observed_data)
    print(f"   Done in {time.perf_counter() - start_time:.4f}s\n")

    print("3. Calculating W2 loss surface via quantile sampling (Method 3 - fast)...")
    start_time = time.perf_counter()
    w2_quant_grid = calculate_w2_quantile_loss_surface(observed_data)
    print(f"   Done in {time.perf_counter() - start_time:.4f}s\n")

    print("Running Powell optimization on W1...")
    powell_res = estimate_w1_powell(observed_data)
    print(f"Powell Optimization result:\n{powell_res}\n")

    # Display heatmaps sequentially (blocking plots)
    print("Displaying W1 Loss Surface. Close plot to continue.")
    brute_heatmap.plot_heatmap(w1_grid, ("mu", "rho"))

    print("Displaying W2 (Random) Loss Surface. Close plot to continue.")
    brute_heatmap.plot_heatmap(w2_rand_grid, ("mu", "rho"))

    print("Displaying W2 (Quantile) Loss Surface. Close plot to finish.")
    brute_heatmap.plot_heatmap(w2_quant_grid, ("mu", "rho"))


if __name__ == "__main__":
    main()
