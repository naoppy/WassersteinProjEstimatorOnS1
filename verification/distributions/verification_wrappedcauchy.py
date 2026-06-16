import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wrapcauchy as sp_wrapcauchy

from src.distributions.wrappedcauchy import (
    MLE_OKAMURA,
    MLE_direct,
    MLE_Kent,
    fisher_info_2x2,
    wrapcauchy_pdf_analytical,
    wrapcauchy_periodic_cdf_analytical,
    wrapcauchy_ppf_analytical,
)
from src.utils.circular_utils import circular_quantile_sampling, to_2pi_range


def test_plot_for_slide() -> None:
    """スライド用の分布描画"""
    print("=== Plotting Wrapped Cauchy (Cartesian/Polar) ===")
    n = 100000
    mu = 0.0
    rho = 0.4
    plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")

    x = np.linspace(0, 2 * np.pi, 1000)
    pdf_vals = wrapcauchy_pdf_analytical(x, rho, mu)

    def ppf_func(q):
        return wrapcauchy_ppf_analytical(q, rho, loc=mu)

    # Use the standardized quantile_sampling
    sample = circular_quantile_sampling(ppf_func, n)

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
    print("=== Testing Wrapped Cauchy Estimators and CDF/PPF ===")
    mu = to_2pi_range(np.pi / 2)  # circular mean in [0, 2*pi]
    rho = 0.7  # concentration
    N = 10000
    dist = sp_wrapcauchy(loc=mu, c=rho)

    # calc Fisher info matrix
    print("Fisher info:")
    print(fisher_info_2x2(rho))

    # scipy outputs in [loc, loc + 2*pi] -> [mu, mu + 2*pi]
    sample = to_2pi_range(dist.rvs(N))
    print(f"Sample range: min={np.min(sample)}, max={np.max(sample)}")

    # calc MLE
    res_okamura = MLE_OKAMURA(sample, N, iter_num=100)
    print(f"mu  MLE by Okamura: {res_okamura[0]}")
    print(f"rho MLE by Okamura: {res_okamura[1]}")

    res_kent = MLE_Kent(sample, debug=True, tol=1e-9)
    print(f"mu  MLE by Kent: {res_kent[0]}")
    print(f"rho MLE by Kent: {res_kent[1]}")

    res_direct = MLE_direct(sample)
    print(f"mu  MLE by direct: {res_direct[0]}")
    print(f"rho MLE by direct: {res_direct[1]}")

    def ppf_func(q):
        return wrapcauchy_ppf_analytical(q, rho, loc=mu)

    sample2 = circular_quantile_sampling(ppf_func, N)
    print(f"Quantile Sample range: min={np.min(sample2)}, max={np.max(sample2)}")

    # plots
    x = np.linspace(0, 2 * np.pi, 1001)
    plt.figure()
    plt.plot(x, wrapcauchy_pdf_analytical(x, rho, mu), label="periodic pdf")
    plt.plot(x, wrapcauchy_periodic_cdf_analytical(x, rho, mu), label="periodic cdf")
    plt.plot(
        x,
        wrapcauchy_periodic_cdf_analytical(x, rho, mu)
        - wrapcauchy_periodic_cdf_analytical(0, rho, mu),
        label="normalized cdf",
    )
    plt.legend()
    plt.title("PDF and CDF Functions")
    plt.show()

    # PPF plot
    x_q = np.linspace(0, 1, 1001)
    y = wrapcauchy_ppf_analytical(x_q, rho, loc=mu)
    y_mod = to_2pi_range(y)
    plt.figure()
    plt.plot(x_q, y, label="ppf")
    plt.plot(x_q, y_mod, label="ppf mod 2pi")
    plt.legend()
    plt.title("PPF Function")
    plt.show()


if __name__ == "__main__":
    test_main()
    test_plot_for_slide()
