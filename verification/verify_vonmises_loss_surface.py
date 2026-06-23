"""Loss surface verification and visualization for von Mises distribution.

This script visualizes the loss landscapes of circular 1-Wasserstein and 2-Wasserstein
distances over the parameters (mu, kappa) of the von Mises distribution.
It demonstrates that the loss surfaces are well-behaved (e.g., unimodal) and suitable
for local optimization methods such as Powell or Nelder-Mead.
"""

from typing import Any, Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from src.distributions import vonmises
from src.method import (
    circular_w1_from_cumsums,
    circular_wasserstein_from_samples,
)
from src.plots import brute_heatmap


def calculate_w1_loss_surface(given_data: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    """Calculate the W1 loss surface (Method 2) over a grid of parameters.

    Args:
        given_data (np.ndarray): Observed circular samples in [-pi, pi].

    Returns:
        Tuple: Optimization result tuple containing:
               [opt_params, opt_val, grid, loss_values]
    """
    bin_num = len(given_data)
    data_cumsum_hist = vonmises.cumsum_hist_data(given_data, bin_num)

    def cost_func(params: np.ndarray) -> float:
        mu, kappa = params
        dist_cumsum_hist = vonmises.cumsum_hist(mu, kappa, bin_num)
        return circular_w1_from_cumsums(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    # Ns=100 grid points for brute force visualization
    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0.01, 10.0)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def calculate_w2_loss_surface(given_data: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    """Calculate the W2 loss surface (Method 1) over a grid of parameters.

    Args:
        given_data (np.ndarray): Observed circular samples in [-pi, pi].

    Returns:
        Tuple: Optimization result tuple containing:
               [opt_params, opt_val, grid, loss_values]
    """
    given_data_norm = np.remainder(given_data, 2 * np.pi) / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)

    def cost_func(params: np.ndarray) -> float:
        mu, kappa = params
        model_sample = stats.vonmises(loc=mu, kappa=kappa).rvs(len(given_data))
        model_sample = np.remainder(model_sample, 2 * np.pi) / (2 * np.pi)
        model_sample = np.sort(model_sample)
        return circular_wasserstein_from_samples(
            given_data_norm_sorted, model_sample, p=2, sorted=True
        )

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0.01, 10.0)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)

    n_samples = 1000  # Reduced for faster interactive visualization
    true_mu = 0.3
    true_kappa = 5.0

    print(
        f"--- von Mises Loss Surface Verification ---"
        f"\nSample Size (N) = {n_samples}"
        f"\nTrue Parameters: mu = {true_mu}, kappa = {true_kappa}\n"
    )

    # Generate synthetic observations (bimodal version to test robustness)
    print("Generating bimodal von Mises observations...")
    sample1 = stats.vonmises(loc=true_mu, kappa=true_kappa).rvs(n_samples // 2)
    sample2 = stats.vonmises(loc=true_mu + np.pi * 0.75, kappa=true_kappa).rvs(
        n_samples - n_samples // 2
    )
    observed_data = np.concatenate([sample1, sample2])

    print(f"MLE Estimate: {vonmises.MLE(vonmises.T(observed_data), n_samples)}")

    print("\n1. Calculating W1 loss surface...")
    w1_grid_results = calculate_w1_loss_surface(observed_data)

    print("\n2. Calculating W2 loss surface (this may take a few seconds)...")
    w2_grid_results = calculate_w2_loss_surface(observed_data)

    # Display heatmaps sequentially (blocking plots)
    print("\nDisplaying W1 Loss Surface Heatmap. Close the plot window to proceed.")
    brute_heatmap.plot_heatmap(w1_grid_results, ("mu", "kappa"))

    print("\nDisplaying W2 Loss Surface Heatmap. Close the plot window to finish.")
    brute_heatmap.plot_heatmap(w2_grid_results, ("mu", "kappa"))


if __name__ == "__main__":
    main()
