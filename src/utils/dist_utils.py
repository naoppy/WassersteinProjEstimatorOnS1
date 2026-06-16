import numba
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


@numba.njit(fastmath=True, parallel=True, cache=True)
def _vonmises_cdf_series_numba(
    theta: npt.NDArray[np.float64], mu: float, r_n: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    n_points = len(theta)
    n_terms = len(r_n)
    sum_vals = np.zeros_like(theta)
    for i in numba.prange(n_points):
        t = theta[i]
        val = 0.0
        for n in range(1, n_terms + 1):
            val += (r_n[n - 1] / n) * (np.sin(n * (t - mu)) + np.sin(n * mu))
        sum_vals[i] = t / (2 * np.pi) + val / np.pi
    return sum_vals


def vonmises_cdf_series(
    theta: npt.NDArray[np.float64], mu: float, kappa: float, n_terms: int = 150
) -> npt.NDArray[np.float64]:
    """Calculates the von Mises CDF starting at 0 on [0, 2*pi]
    using Fourier-Bessel series.

    This implementation guarantees F(0) = 0.

    Note:
        As kappa (concentration) increases, more terms (n_terms) are required
        for the series to converge to double precision (1e-16).
        The following table shows the approximate number of terms required
        to reach machine precision:

        kappa   | Required n_terms
        --------|-----------------
        1.0     | 15
        10.0    | 32
        50.0    | 64
        100.0   | 88
        250.0   | 150

    """
    from scipy.special import ive

    theta_arr = np.asarray(theta, dtype=np.float64)
    n_arr = np.arange(1, n_terms + 1)
    r_n = ive(n_arr, kappa) / ive(0, kappa)

    orig_shape = theta_arr.shape
    theta_flat = theta_arr.ravel()
    res = _vonmises_cdf_series_numba(theta_flat, mu, r_n).reshape(orig_shape)
    if res.ndim == 0:
        return float(res)
    return res


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
