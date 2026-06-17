import numpy as np
from scipy.interpolate import interp1d


def get_interpolated_ppf(cdf_func, grid_size: int = 2000):
    """Generates an interpolated PPF function on [0, 1]."""
    grid = np.linspace(0, 2 * np.pi, grid_size)
    cdf_vals = cdf_func(grid)
    eps = 1e-12
    cdf_vals_strict = cdf_vals + np.arange(len(cdf_vals)) * eps
    cdf_vals_strict = (cdf_vals_strict - cdf_vals_strict[0]) / (
        cdf_vals_strict[-1] - cdf_vals_strict[0]
    )
    return interp1d(
        cdf_vals_strict,
        grid / (2 * np.pi),
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )


def kl_vonmises_wrapcauchy_analytical(
    mu_p: float, kappa: float, mu_q: float, rho: float, n_terms: int = 150
) -> float:
    """Calculates the analytical KL divergence D_KL(P || Q) where
    P is von Mises(mu_p, kappa) and Q is wrapped Cauchy(mu_q, rho).
    """
    from scipy.special import ive

    r1 = ive(1, kappa) / ive(0, kappa)
    log_i0 = np.log(ive(0, kappa)) + kappa

    n_arr = np.arange(1, n_terms + 1)
    r_n = ive(n_arr, kappa) / ive(0, kappa)

    cos_term = np.cos(n_arr * (mu_p - mu_q))
    series_sum = np.sum((2.0 * (rho**n_arr) * r_n / n_arr) * cos_term)

    kl = kappa * r1 - log_i0 - np.log(1.0 - rho**2) - series_sum
    return float(kl)


def kl_wrapcauchy_vonmises_analytical(
    mu_q: float, rho: float, mu_p: float, kappa: float
) -> float:
    """Calculates the closed-form KL divergence D_KL(Q || P) where
    Q is wrapped Cauchy(mu_q, rho) and P is von Mises(mu_p, kappa).
    """
    from scipy.special import ive

    log_i0 = np.log(ive(0, kappa)) + kappa
    kl = log_i0 - np.log(1.0 - rho**2) - kappa * rho * np.cos(mu_q - mu_p)
    return float(kl)


def vm_mean_abs_dev(kappa: float, n_terms: int = 150) -> float:
    """Calculates the Mean Absolute Deviation E_P[|theta|] for von Mises
    (mean 0) on [-pi, pi].
    """
    from scipy.special import ive

    k_arr = np.arange(0, n_terms)
    n_arr = 2 * k_arr + 1
    r_n = ive(n_arr, kappa) / ive(0, kappa)
    series_sum = np.sum(r_n / (n_arr**2))
    return float(np.pi / 2.0 - (4.0 / np.pi) * series_sum)


def wc_mean_abs_dev(rho: float, n_terms: int = 150) -> float:
    """Calculates the Mean Absolute Deviation E_Q[|theta|] for wrapped Cauchy
    (mean 0) on [-pi, pi].
    """
    k_arr = np.arange(0, n_terms)
    n_arr = 2 * k_arr + 1
    series_sum = np.sum((rho**n_arr) / (n_arr**2))
    return float(np.pi / 2.0 - (4.0 / np.pi) * series_sum)


def w1_aligned_analytical(kappa: float, rho: float, n_terms: int = 150) -> float:
    """Calculates the analytical W1 distance when the mean directions are aligned
    (mu_p = mu_q) and one distribution is more concentrated than the other.
    """
    evm = vm_mean_abs_dev(kappa, n_terms)
    ewc = wc_mean_abs_dev(rho, n_terms)
    return abs(evm - ewc)


def calculate_distances_vonmises_wrappedcauchy(
    mu_vM: float,
    kappa: float,
    mu_WC: float,
    rho: float,
    grid_size_w1: int = 10000,
    grid_size_w2: int = 5000,
    ppf_interp_grid_size: int = 5000,
) -> tuple[float, float, float, float]:
    """Calculates analytical KL divergences (both VM||WC and WC||VM),
    W1 distance, and W2 distance using 1D continuous optimization.

    Args:
        mu_vM: von Mises mean direction.
        kappa: von Mises concentration parameter.
        mu_WC: wrapped Cauchy mean direction.
        rho: wrapped Cauchy concentration parameter.
        grid_size_w1: Grid size for W1 computation.
        grid_size_w2: Grid size for W2 computation.
        ppf_interp_grid_size: Grid size for building VM PPF interpolation.

    Returns:
        tuple: (kl_vm_wc, kl_wc_vm, w1, w2)
    """
    from scipy.optimize import minimize_scalar

    from src.distributions.vonmises import vonmises_cdf_series
    from src.distributions.wrappedcauchy import (
        wrapcauchy_periodic_cdf_analytical,
        wrapcauchy_ppf_analytical,
    )

    # 1. KL divergences
    kl_vm_wc = kl_vonmises_wrapcauchy_analytical(mu_vM, kappa, mu_WC, rho)
    kl_wc_vm = kl_wrapcauchy_vonmises_analytical(mu_WC, rho, mu_vM, kappa)

    # 2. W1 distance
    def p_cdf(theta):
        return vonmises_cdf_series(theta, mu_vM, kappa)

    t_grid_w1 = np.linspace(1e-9, 1 - 1e-9, grid_size_w1)
    g_p_vals = p_cdf(2 * np.pi * t_grid_w1)
    g_q_vals = wrapcauchy_periodic_cdf_analytical(
        2 * np.pi * t_grid_w1, rho, mu_WC
    ) - wrapcauchy_periodic_cdf_analytical(0, rho, mu_WC)

    diffs_w1 = g_p_vals - g_q_vals
    c_opt = np.median(diffs_w1)
    w1 = 2 * np.pi * np.mean(np.abs(diffs_w1 - c_opt))

    # 3. W2 distance
    p_ppf_norm = get_interpolated_ppf(p_cdf, grid_size=ppf_interp_grid_size)

    def p_ppf(q):
        return p_ppf_norm(q) * 2 * np.pi

    def q_ppf(q):
        return wrapcauchy_ppf_analytical(q, rho, loc=mu_WC)

    def p_ppf_extended(u):
        u_mod = np.remainder(u, 1.0)
        periods = np.floor(u)
        return p_ppf(u_mod) + periods * 2 * np.pi

    def q_ppf_extended(u):
        u_mod = np.remainder(u, 1.0)
        periods = np.floor(u)
        return q_ppf(u_mod) + periods * 2 * np.pi

    u_grid = np.linspace(1e-9, 1 - 1e-9, grid_size_w2)
    p_vals = p_ppf_extended(u_grid)

    def loss_fn(alpha):
        q_vals = q_ppf_extended(u_grid + alpha)
        return np.mean((p_vals - q_vals) ** 2)

    res = minimize_scalar(loss_fn, bounds=(-1.5, 1.5), method="bounded")
    w2 = np.sqrt(res.fun)

    return kl_vm_wc, kl_wc_vm, w1, w2
