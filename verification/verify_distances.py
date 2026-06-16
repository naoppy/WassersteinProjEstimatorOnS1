import time

import numpy as np
import ot
import scipy.stats as stats

from src.distributions import wrappedcauchy
from src.distributions.vonmises import vonmises_pdf_stable
from src.distributions.wrappedcauchy import wrapcauchy_pdf_analytical
from src.utils import dist_utils


def main():
    print("=== Verification of Circular Distance Functions ===")

    # 1. Test W1/W2 between concentrated distributions with varying offset (geodesic)
    print("\n1. Testing geodesic distance properties:")
    kappa = 5000.0  # act like Dirac masses
    offsets = [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2]

    for offset in offsets:
        mu1 = 0.5
        mu2 = np.remainder(mu1 + offset, 2 * np.pi)

        # True geodesic distance on the circle
        true_geodesic = np.minimum(np.abs(offset), 2 * np.pi - np.abs(offset))
        if offset < 0:
            true_geodesic = np.minimum(
                np.abs(offset + 2 * np.pi), 2 * np.pi - np.abs(offset + 2 * np.pi)
            )

        def p_pdf(theta, mu1=mu1, kappa=kappa):
            return vonmises_pdf_stable(theta, mu1, kappa)

        def q_pdf(theta, mu2=mu2, kappa=kappa):
            return vonmises_pdf_stable(theta, mu2, kappa)

        dist_p = stats.vonmises(loc=mu1, kappa=kappa)
        dist_q = stats.vonmises(loc=mu2, kappa=kappa)

        def p_cdf(theta, dist_p=dist_p):
            return dist_p.cdf(theta) - dist_p.cdf(0)

        def q_cdf(theta, dist_q=dist_q):
            return dist_q.cdf(theta) - dist_q.cdf(0)

        # Clip to prevent infs/NaNs at exactly 0 and 1
        def p_ppf(q, dist_p=dist_p):
            return dist_p.ppf(np.clip(q, 1e-12, 1 - 1e-12))

        def q_ppf(q, dist_q=dist_q):
            return dist_q.ppf(np.clip(q, 1e-12, 1 - 1e-12))

        kl, w1, w2 = dist_utils.calculate_distances(
            p_pdf, q_pdf, p_cdf=p_cdf, q_cdf=q_cdf, p_ppf=p_ppf, q_ppf=q_ppf
        )

        print(
            f"Offset: {offset:.3f} | True Geodesic: {true_geodesic:.3f} | "
            f"Computed W1: {w1:.3f} | Computed W2: {w2:.3f}"
        )
        assert np.abs(w1 - true_geodesic) < 0.1, f"W1 mismatch: {w1} vs {true_geodesic}"
        assert np.abs(w2 - true_geodesic) < 0.1, f"W2 mismatch: {w2} vs {true_geodesic}"

    # 2. Test W2 between von Mises and Wrapped Cauchy against POT ground-truth samples
    print("\n2. Comparing calculate_distances (PPF-based) vs POT sample ground-truth:")
    mu1, kappa = 0.1, 100.0
    mu2, rho = 1.1, 0.75

    def p_pdf_check(theta, mu1=mu1, kappa=kappa):
        return vonmises_pdf_stable(theta, mu1, kappa)

    def q_pdf_check(theta, mu2=mu2, rho=rho):
        return wrapcauchy_pdf_analytical(theta, rho, loc=mu2)

    dist_p = stats.vonmises(loc=mu1, kappa=kappa)

    def p_cdf_check(theta, dist_p=dist_p):
        return dist_p.cdf(theta) - dist_p.cdf(0)

    def p_ppf_check(q, dist_p=dist_p):
        return dist_p.ppf(np.clip(q, 1e-12, 1 - 1e-12))

    def q_cdf_check(theta, rho=rho, mu2=mu2):
        return wrappedcauchy.wrapcauchy_periodic_cdf_analytical(
            theta, rho, mu2
        ) - wrappedcauchy.wrapcauchy_periodic_cdf_analytical(0, rho, mu2)

    def q_ppf_check(q, rho=rho, mu2=mu2):
        return wrappedcauchy.wrapcauchy_ppf_analytical(q, rho, loc=mu2)

    # New method
    kl, w1, w2 = dist_utils.calculate_distances(
        p_pdf_check,
        q_pdf_check,
        p_cdf=p_cdf_check,
        q_cdf=q_cdf_check,
        p_ppf=p_ppf_check,
        q_ppf=q_ppf_check,
    )

    # Ground-truth samples using POT
    n_samples = 2000
    p_samples = dist_p.rvs(n_samples)
    q_samples = wrappedcauchy.wrapcauchy_ppf_analytical(
        np.random.rand(n_samples), rho, loc=mu2
    )
    p_samples_norm = np.remainder(p_samples, 2 * np.pi) / (2 * np.pi)
    q_samples_norm = np.remainder(q_samples, 2 * np.pi) / (2 * np.pi)

    gt_w2_sq = ot.binary_search_circle(p_samples_norm, q_samples_norm, p=2)
    gt_w2 = 2 * np.pi * np.sqrt(gt_w2_sq[0])

    print(f"Computed W2: {w2:.6f}")
    print(f"POT Sample W2: {gt_w2:.6f}")
    assert np.abs(w2 - gt_w2) < 0.05, f"W2 distance mismatch: {w2} vs {gt_w2}"
    print("Success! PPF-based W2 matches sample-based POT W2.")

    # 3. Speed benchmark for W1 distance (Median vs minimize_scalar)
    print("\n3. W1 computation time test:")
    from scipy import optimize

    t_grid_w1 = np.linspace(1e-9, 1 - 1e-9, 1000)
    g_p_vals = p_cdf_check(2 * np.pi * t_grid_w1)
    g_q_vals = q_cdf_check(2 * np.pi * t_grid_w1)

    # Old optimizer-based method
    start = time.perf_counter()
    for _ in range(100):

        def w1_loss(c):
            return np.mean(np.abs(g_p_vals - g_q_vals - c))

        res = optimize.minimize_scalar(w1_loss, bounds=(-1, 1), method="bounded")
        w1_old = 2 * np.pi * res.fun
    old_time = (time.perf_counter() - start) / 100

    # New median-based method
    start = time.perf_counter()
    for _ in range(100):
        diffs_w1 = g_p_vals - g_q_vals
        c_opt = np.median(diffs_w1)
        w1_new = 2 * np.pi * np.mean(np.abs(diffs_w1 - c_opt))
    new_time = (time.perf_counter() - start) / 100

    print(f"Old method time: {old_time * 1000:.3f} ms | value: {w1_old:.6f}")
    print(f"New method time: {new_time * 1000:.3f} ms | value: {w1_new:.6f}")
    print(f"Speedup: {old_time / new_time:.1f}x")
    assert np.abs(w1_old - w1_new) < 1e-12, "W1 values mismatch"


if __name__ == "__main__":
    main()
