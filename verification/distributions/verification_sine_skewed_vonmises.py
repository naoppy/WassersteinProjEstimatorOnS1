import numpy as np
from matplotlib import pyplot as plt

from src.distributions.sine_skewed_vonmises import (
    MLE_direct,
    fisher_info_3x3,
    fisher_mat_inv_diag,
    rejection_sampling,
)
from src.distributions.sine_skewed_vonmises import (
    sine_skewed_vonmises_pdf_analytical as pdf,
)


def test_plot_for_slide() -> None:
    """スライド用の分布描画"""
    print("=== Plotting Sine-Skewed von Mises (Cartesian/Polar) ===")
    mu = 0.0
    kappa = 1.0
    lambda_ = 0.7
    plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")

    x = np.linspace(0, 2 * np.pi, 1000)
    ss_vM_pdf = pdf(x, mu, kappa, lambda_)

    # Map to [-pi, pi] for display to match the original slide format
    x_display = np.remainder(x + np.pi, 2 * np.pi) - np.pi
    # Re-sort for continuous line plotting
    sort_idx = np.argsort(x_display)
    x_display = x_display[sort_idx]
    ss_vM_pdf = ss_vM_pdf[sort_idx]

    ticks = [0, 0.15, 0.3]

    left.plot(x_display, ss_vM_pdf)
    left.set_yticks(ticks)
    left.fill_between(x_display, ss_vM_pdf, color="tab:orange", alpha=0.6)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x_display, ss_vM_pdf, label="PDF")
    right.set_yticks(ticks)
    right.fill_between(
        x_display, ss_vM_pdf, color="tab:orange", alpha=0.6, label="PDF fill"
    )
    right.set_title("Polar plot")

    right.legend(bbox_to_anchor=(0.15, 1.06))
    plt.tight_layout()
    plt.show()


def test_main() -> None:
    """Main verification logic"""
    print("=== Testing Sine-Skewed von Mises Estimators ===")
    n = 100000
    kappa = 1.0
    lambda_ = 0.7
    mu = 0.0
    sample = rejection_sampling(n, mu, kappa, lambda_, debug=True)
    print(f"Sample range: min={np.min(sample)}, max={np.max(sample)}")

    plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")
    x = np.linspace(0, 2 * np.pi, 1000)
    ss_vonmises_pdf = pdf(x, mu, kappa, lambda_)

    # Map to [-pi, pi] for display
    x_display = np.remainder(x + np.pi, 2 * np.pi) - np.pi
    sample_display = np.remainder(sample + np.pi, 2 * np.pi) - np.pi
    sort_idx = np.argsort(x_display)
    x_display = x_display[sort_idx]
    ss_vonmises_pdf = ss_vonmises_pdf[sort_idx]

    ticks = [0, 0.15, 0.3]

    left.plot(x_display, ss_vonmises_pdf)
    left.set_yticks(ticks)
    number_of_bins = int(np.sqrt(n))
    left.hist(sample_display, density=True, bins=number_of_bins, alpha=0.6)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x_display, ss_vonmises_pdf, label="PDF")
    right.set_yticks(ticks)
    right.hist(
        sample_display, density=True, bins=number_of_bins, alpha=0.6, label="Histogram"
    )
    right.set_title("Polar plot")

    # param estimation
    est_param = MLE_direct(sample, debug=True)
    print(
        f"Estimated parameters: mu={est_param[0]:.4f}, "
        f"kappa={est_param[1]:.4f}, lambda={est_param[2]:.4f}"
    )
    ss_vonmises_est_pdf = pdf(x, est_param[0], est_param[1], est_param[2])
    # Map to [-pi, pi] for display
    ss_vonmises_est_pdf = ss_vonmises_est_pdf[sort_idx]

    left.plot(x_display, ss_vonmises_est_pdf, label="Estimated PDF", linestyle="--")
    right.plot(x_display, ss_vonmises_est_pdf, label="Estimated PDF", linestyle="--")

    right.legend(bbox_to_anchor=(0.15, 1.06))
    plt.tight_layout()
    plt.show()

    mat = fisher_info_3x3(kappa, lambda_)
    print("Fisher info:")
    print(mat)
    mat_inv_diag = fisher_mat_inv_diag(kappa, lambda_)
    print("Fisher info inverse diagonal:")
    print(mat_inv_diag)


if __name__ == "__main__":
    test_main()
    test_plot_for_slide()
