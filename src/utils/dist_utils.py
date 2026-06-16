import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d


def vonmises_pdf(
    theta: npt.NDArray[np.float64], mu: float, kappa: float
) -> npt.NDArray[np.float64]:
    """Stable von Mises PDF using stable vonmises module."""
    from src.distributions import vonmises

    return vonmises.vonmises_pdf_stable(theta, mu, kappa)


def wrapcauchy_pdf(
    theta: npt.NDArray[np.float64], mu: float, rho: float
) -> npt.NDArray[np.float64]:
    """Wrapped Cauchy PDF using stable wrappedcauchy module."""
    from src.distributions import wrappedcauchy

    return wrappedcauchy.wrapcauchy_pdf_analytical(theta, rho, mu)


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


def calculate_distances(p_pdf, q_pdf, p_cdf=None, q_cdf=None, p_ppf=None, q_ppf=None):
    """Calculates KL, W1, and W2 distances between P and Q distributions on S1."""
    import ot

    # KL divergence
    def kl_integrand(theta):
        val_p = p_pdf(theta)
        val_q = q_pdf(theta)
        if val_p <= 0:
            return 0.0
        val_q = max(val_q, 1e-300)
        return val_p * (np.log(val_p) - np.log(val_q))

    kl_div, _ = quad(kl_integrand, 0, 2 * np.pi, limit=100)

    # CDF representations
    cdf_P = p_cdf if p_cdf is not None else get_interpolated_cdf(p_pdf)
    cdf_Q = q_cdf if q_cdf is not None else get_interpolated_cdf(q_pdf)

    # W1 distance
    t_grid_w1 = np.linspace(1e-9, 1 - 1e-9, 1000)
    g_p_vals = cdf_P(2 * np.pi * t_grid_w1)
    g_q_vals = cdf_Q(2 * np.pi * t_grid_w1)
    diffs_w1 = g_p_vals - g_q_vals
    c_opt = np.median(diffs_w1)
    w1 = 2 * np.pi * np.mean(np.abs(diffs_w1 - c_opt))

    # W2 distance
    if p_ppf is not None:
        ppf_P_vals = p_ppf(t_grid_w1)
    else:
        ppf_P_vals = get_interpolated_ppf(cdf_P)(t_grid_w1) * 2 * np.pi

    if q_ppf is not None:
        ppf_Q_vals = q_ppf(t_grid_w1)
    else:
        ppf_Q_vals = get_interpolated_ppf(cdf_Q)(t_grid_w1) * 2 * np.pi

    # Normalize PPF values to [0, 1] range for POT binary_search_circle
    p_vals_norm = np.remainder(ppf_P_vals, 2 * np.pi) / (2 * np.pi)
    q_vals_norm = np.remainder(ppf_Q_vals, 2 * np.pi) / (2 * np.pi)

    # ot.binary_search_circle returns W_2^2, so we take the square root
    w2_sq = ot.binary_search_circle(
        p_vals_norm, q_vals_norm, p=2, log=False, require_sort=True
    )
    if isinstance(w2_sq, np.ndarray):
        w2_sq = w2_sq[0]
    w2 = 2 * np.pi * np.sqrt(w2_sq)

    return kl_div, w1, w2
