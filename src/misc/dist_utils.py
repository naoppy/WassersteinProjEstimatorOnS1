import numpy as np
import numpy.typing as npt
import scipy.special as special
from scipy import optimize
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d


def vonmises_pdf(
    theta: npt.NDArray[np.float64], mu: float, kappa: float
) -> npt.NDArray[np.float64]:
    """Stable von Mises PDF using ive to prevent overflow."""
    return np.exp(kappa * (np.cos(theta - mu) - 1)) / (
        2 * np.pi * special.ive(0, kappa)
    )


def wrapcauchy_pdf(
    theta: npt.NDArray[np.float64], mu: float, rho: float
) -> npt.NDArray[np.float64]:
    """Wrapped Cauchy PDF using stable wrapedcauchy module."""
    from ..distributions import wrapedcauchy

    return wrapedcauchy.wrapcauchy_pdf_analytical(theta, rho, mu)


def get_interpolated_cdf(pdf_func, grid_size: int = 2000):
    """Generates an interpolated CDF function on [0, 2*pi]."""
    grid = np.linspace(0, 2 * np.pi, grid_size)
    pdf_vals = pdf_func(grid)
    cdf_vals = cumulative_trapezoid(pdf_vals, grid, initial=0)
    cdf_vals /= cdf_vals[-1]
    return interp1d(
        grid,
        cdf_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )


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


def calculate_distances(p_pdf, q_pdf, p_cdf=None, q_cdf=None):
    """Calculates KL, W1, and W2 distances between P and Q distributions on S1."""

    # KL divergence
    def kl_integrand(theta):
        val_p = p_pdf(theta)
        val_q = q_pdf(theta)
        if val_p <= 0:
            return 0.0
        val_q = max(val_q, 1e-300)
        return val_p * (np.log(val_p) - np.log(val_q))

    kl_div, _ = quad(kl_integrand, 0, 2 * np.pi, limit=100)

    # CDF and PPF representations
    cdf_P = p_cdf if p_cdf is not None else get_interpolated_cdf(p_pdf)
    cdf_Q = q_cdf if q_cdf is not None else get_interpolated_cdf(q_pdf)

    # W1 distance
    def g_p(t):
        return cdf_P(2 * np.pi * t)

    def g_q(t):
        return cdf_Q(2 * np.pi * t)

    t_grid_w1 = np.linspace(0, 1, 1000)
    g_p_vals = g_p(t_grid_w1)
    g_q_vals = g_q(t_grid_w1)

    def w1_loss(c):
        return np.mean(np.abs(g_p_vals - g_q_vals - c))

    res = optimize.minimize_scalar(w1_loss, bounds=(-1, 1), method="bounded")
    w1 = 2 * np.pi * res.fun

    # W2 distance
    ppf_P = get_interpolated_ppf(cdf_P)
    ppf_Q = get_interpolated_ppf(cdf_Q)

    t_grid = np.linspace(0, 1, 2000)
    diffs = ppf_P(t_grid) - ppf_Q(t_grid)
    mean_diff = np.mean(diffs)
    var_diff = np.mean((diffs - mean_diff) ** 2)
    w2 = 2 * np.pi * np.sqrt(var_diff)

    return kl_div, w1, w2
