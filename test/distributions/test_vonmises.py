import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import vonmises

from src.distributions.vonmises import (
    MLE_direct,
    fisher_info_2x2,
    quantile_sampling,
    type0_estimate,
    type1_estimate,
    vonmises_pdf_stable,
)
from src.misc.circular_utils import to_2pi_range


def test_estimate() -> None:
    """いろんな推定量を計算してみる"""
    print("=== Testing Parameter Estimators for von Mises ===")
    mu = to_2pi_range(0.5 * np.pi)  # 0.5 * pi
    kappa = 1.3  # concentration
    N = 10000
    dist = vonmises(loc=mu, kappa=kappa)
    # scipy vonmises outputs samples in [mu - pi, mu + pi] by default.
    # Map them to [0, 2*pi] standard.
    sample = to_2pi_range(dist.rvs(N))

    # MLE_direct expects a sample in [0, 2*pi]
    mu_MLE, kappa_MLE = MLE_direct(sample)
    print(f"MLE: mu={mu_MLE}, kappa={kappa_MLE}")

    # type0_estimate and type1_estimate
    mu_type0, kappa_type0 = type0_estimate(sample, gamma=0.0, debug=False)
    print(f"type0 estimator (gamma=0): mu={mu_type0}, kappa={kappa_type0}")

    mu_type1, kappa_type1 = type1_estimate(sample, beta=0.0, debug=False)
    print(f"type1 estimator (beta=0): mu={mu_type1}, kappa={kappa_type1}")

    mu_type0_g, kappa_type0_g = type0_estimate(sample, gamma=0.5, debug=False)
    print(f"type0 estimator (gamma=0.5): mu={mu_type0_g}, kappa={kappa_type0_g}")

    mu_type1_b, kappa_type1_b = type1_estimate(sample, beta=0.5, debug=False)
    print(f"type1 estimator (beta=0.5): mu={mu_type1_b}, kappa={kappa_type1_b}")


def test_plot_for_slide() -> None:
    """スライドに載せる分布の例の画像を作成する"""
    print("=== Plotting Distribution (Cartesian/Polar) ===")
    n = 100000
    mu = 0.0
    kappa = 2.0
    plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")

    # Define x-range standard [0, 2*pi]
    x = np.linspace(0, 2 * np.pi, 1000)
    pdf_vals = vonmises_pdf_stable(x, mu, kappa)

    # Use the standardized quantile_sampling
    sample = quantile_sampling(mu, kappa, n)

    # Map to [-pi, pi] for display to match the original slide format
    x_display = np.remainder(x + np.pi, 2 * np.pi) - np.pi
    sample_display = np.remainder(sample + np.pi, 2 * np.pi) - np.pi
    # Re-sort for continuous line plotting
    sort_idx = np.argsort(x_display)
    x_display = x_display[sort_idx]
    pdf_vals = pdf_vals[sort_idx]

    ticks = [0, 0.15, 0.3]

    left.plot(x_display, pdf_vals, label="PDF")
    left.set_yticks(ticks)
    number_of_bins = int(np.sqrt(n))
    left.hist(sample_display, density=True, bins=number_of_bins, alpha=0.6)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x_display, pdf_vals, label="PDF")
    right.set_yticks(ticks)
    right.hist(
        sample_display, density=True, bins=number_of_bins, alpha=0.6, label="Histogram"
    )
    right.set_title("Polar plot")

    right.legend(bbox_to_anchor=(0.15, 1.06))
    plt.tight_layout()
    plt.show()


def test_main() -> None:
    """Main verification logic"""
    mu = to_2pi_range(0.5 * np.pi)
    kappa = 1.3
    N = 10000

    print("Fisher info:")
    print(fisher_info_2x2(kappa))

    dist = vonmises(loc=mu, kappa=kappa)
    sample = to_2pi_range(dist.rvs(N))
    print(f"Sample range: min={np.min(sample)}, max={np.max(sample)}")

    mu_MLE, kappa_MLE = MLE_direct(sample)
    print(f"MLE: mu={mu_MLE}, kappa={kappa_MLE}")

    sample2 = quantile_sampling(mu, kappa, N)
    print(f"Quantile Sample range: min={np.min(sample2)}, max={np.max(sample2)}")

    # Plot verification
    x = np.linspace(0, 2 * np.pi, 1001)
    plt.figure()
    plt.plot(x, vonmises_pdf_stable(x, mu, kappa), label="pdf")
    plt.plot(x, dist.cdf(x) - dist.cdf(0), label="normalized cdf")
    plt.legend()
    plt.title("PDF and Normalized CDF")
    plt.show()


if __name__ == "__main__":
    test_estimate()
    test_main()
    test_plot_for_slide()
