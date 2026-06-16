from functools import partial
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from scipy import optimize

from src.method.wasserstein import (
    circular_w1_from_cumsums,
    circular_wasserstein_from_samples,
)
from src.utils.circular_utils import (
    circular_quantile_sampling,
    cumsum_hist_data,
    to_2pi_range,
)

bounds = ((-np.pi, np.pi), (0.01, 0.99))


def fisher_info_2x2(rho: float) -> npt.NDArray[np.float64]:
    """巻き込みコーシー分布のフィッシャー情報量を計算する"""
    rho_p2 = rho * rho
    bunbo = (1 - rho_p2) ** 2
    return np.array(
        [
            [2 * rho_p2 / bunbo, 0],
            [0, 2 / bunbo],
        ]
    )


def fisher_mat_inv_diag(rho: float) -> List[float]:
    """巻き込みコーシー分布のフィッシャー情報量の逆行列の対角成分を計算する"""
    rho_p2 = rho * rho
    bunbo = (1 - rho_p2) ** 2
    return [bunbo / (2 * rho_p2), bunbo / 2]


def wrapcauchy_pdf_analytical(
    theta: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """巻き込みコーシー分布のPDF値を計算する。数値的に安定な式を使用。"""
    d = (theta - loc) / scale
    denom = (1.0 - c) ** 2 + 4.0 * c * np.sin(d / 2.0) ** 2
    denom = np.clip(denom, 1e-30, None)
    return (1.0 - c**2) / (2.0 * np.pi * scale * denom)


def wrapcauchy_periodic_cdf_analytical(
    x: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """巻き込みコーシー分布の累積分布関数を計算する。
    R全体で単調増加し、F(x + 2*pi*scale) = F(x) + 1 かつ F(0) = 0 となるように定義。
    """
    x = np.asarray(x)

    def cdf_raw(val):
        norm_val = np.remainder(val - loc, 2.0 * np.pi * scale) / scale
        factor = (1.0 + c) / (1.0 - c)
        res = np.empty_like(norm_val, dtype=np.float64)

        mask1 = norm_val < np.pi
        res[mask1] = (1.0 / np.pi) * np.arctan(factor * np.tan(norm_val[mask1] / 2.0))

        mask2 = norm_val == np.pi
        res[mask2] = 0.5

        mask3 = norm_val > np.pi
        res[mask3] = 1.0 - (1.0 / np.pi) * np.arctan(
            factor * np.tan((2.0 * np.pi - norm_val[mask3]) / 2.0)
        )

        return res + np.floor_divide(val - loc, 2.0 * np.pi * scale)

    val_mod = np.remainder(x, 2.0 * np.pi * scale)
    raw_val = cdf_raw(val_mod) - cdf_raw(0.0)
    periods = np.floor_divide(x, 2.0 * np.pi * scale)
    return raw_val + periods


def wrapcauchy_ppf_analytical(
    q: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """巻き込みコーシー分布の分位関数を計算する。数値的に安定な式を使用。"""
    q = np.asarray(q)
    factor = (1.0 - c) / (1.0 + c)
    res = np.empty_like(q, dtype=np.float64)

    mask1 = q < 0.5
    res[mask1] = 2.0 * np.arctan(factor * np.tan(np.pi * q[mask1]))

    mask2 = q == 0.5
    res[mask2] = np.pi

    mask3 = q > 0.5
    res[mask3] = 2.0 * np.pi - 2.0 * np.arctan(
        factor * np.tan(np.pi * (1.0 - q[mask3]))
    )

    return loc + res * scale


def cumsum_hist(mu: float, rho: float, bin_num: int) -> npt.NDArray[np.float64]:
    """[0, 2pi] を bin_num 等分した区間でのcdfの値を返す"""
    x = np.linspace(0, 2 * np.pi, bin_num + 1)
    y = wrapcauchy_periodic_cdf_analytical(x, rho, mu, 1)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def MLE_OKAMURA(x, N: int, iter_num: int = 100) -> npt.NDArray[np.float64]:
    """Okamura et al. (2021) のアルゴリズムによる最尤推定。

    CHARACTERIZATIONS OF THE MAXIMUM LIKELIHOOD ESTIMATOR OF THE CAUCHY DISTRIBUTION
    https://arxiv.org/abs/2104.06130
    で提案されているコーシー分布に対する最尤推定法を実装する。
    指数的に収束する反復法で、MLEに必ず収束する。

    Args:
        x: 0~2piの角度データ
        N (int): データ数
        iter_num (int, optional): 反復回数. Defaults to 100.

    Returns:
        npt.NDArray[np.float64]: [mu, rho]。muは [-pi, pi] の範囲。rhoは [0, 1] の範囲。
    """
    if len(x) != N:
        raise ValueError("The length of x must be equal to N")
    if N < 3:
        raise ValueError("N must be greater than or equal to 3")
    x = np.array(x)

    def _q(w, n, x_val):
        return n / (np.sum(1 / (np.exp(1j * x_val) - 1 / w))) + 1 / w

    my_q = partial(_q, n=N, x_val=x)

    def my_Q(theta):
        return my_q(my_q(theta))

    v = 1 / 2 + 1j / 2
    for _ in range(iter_num):
        v = my_Q(v)
    return np.array([np.angle(v), np.abs(v)])


def MLE_Kent(
    x: npt.NDArray[np.float64],
    tol: float = 1e-15,
    max_iter: int = 10000,
    debug: bool = False,
) -> npt.NDArray[np.float64]:
    """Kent and Tyler (1988) の固定点反復法による最尤推定。

    Maximum Likelihood Estimation for Wrapped Cauchy Distribution, Kent and Tyler, 1988

    Args:
        x (npt.NDArray[np.float64]): 0~2piの角度データ
        tol (float, optional): 収束判定の閾値. Defaults to 1e-15.
        max_iter (int, optional): 最大反復回数. Defaults to 10000.
        debug (bool, optional): デバッグモード. Defaults to False.

    Returns:
        npt.NDArray[np.float64]: [mu_MLE, rho_MLE]。
            muは [-pi, pi] の範囲。rhoは [0, 1] の範囲。
    """
    # N = len(x)
    x = to_2pi_range(x)
    y = np.array([np.cos(x), np.sin(x)])  # (2, N)
    eta = np.array([0.5, 0.5])  # (2,) 初期値
    for i in range(max_iter):
        w = 1 / (1 - eta @ y)  # (N,)
        eta_new = np.sum(w * y, axis=1) / np.sum(w)  # (2,)
        if np.linalg.norm(eta_new - eta) < tol:
            if debug:
                print(f"wrapcauchy kent MLE: Converged at {i}th iteration")
            eta = eta_new
            break
        eta = eta_new
    mu = np.arctan2(eta[1], eta[0])
    eta_norm_pow2 = eta @ eta
    if eta_norm_pow2 < 1e-20:
        rho = 0.0
    else:
        eta_norm_pow2 = np.clip(eta_norm_pow2, 0.0, 1.0 - 1e-15)
        rho = (1.0 - np.sqrt(1.0 - eta_norm_pow2)) / np.sqrt(eta_norm_pow2)
    return np.array([mu, rho])


def neg_log_likelihood(params, data: npt.NDArray[np.float64]) -> float:
    """負の対数尤度関数"""
    mu, rho = params
    pdf_vals = wrapcauchy_pdf_analytical(data, rho, mu)
    eps = 1e-10
    log_likelihood = np.sum(np.log(np.clip(pdf_vals, eps, None)))
    return -log_likelihood  # 最小化関数用にマイナスを返す


def MLE_direct(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """最適化による最尤推定"""
    x = to_2pi_range(x)
    result = optimize.minimize(
        neg_log_likelihood,
        (0, 0.5),
        args=(x,),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )
    return np.array([result.x[0], result.x[1]])


def W1_equal_div_cost_func(
    x, bin_num: int, data_cumsum_hist: npt.NDArray[np.float64]
) -> float:
    mu, rho = x
    dist_cumsum_hist = cumsum_hist(mu, rho, bin_num)
    return circular_w1_from_cumsums(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def W1_equal_div(
    given_data: npt.NDArray[np.float64],
    x0: Optional[npt.NDArray[np.float64]] = None,
    method="powell",
) -> optimize.OptimizeResult:
    """1-Wasserstein 距離（等分割ヒストグラム）を最小化するパラメータ推定"""
    given_data = to_2pi_range(given_data)
    bin_num = len(given_data)
    data_cumsum_hist = cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        W1_equal_div_cost_func, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    if method == "differential_evolution":
        return optimize.differential_evolution(cost_func, bounds=bounds)
    else:
        if x0 is None:
            raise ValueError("x0 is required for local minimization")
        return optimize.minimize(
            cost_func,
            x0,
            bounds=bounds,
            method=method,
            options={"xtol": 1e-6, "ftol": 1e-6},
        )


def W1_quantile_sampling_cost_func(
    x, given_data_normed_sorted: npt.NDArray[np.float64]
) -> float:
    def ppf_func(q):
        return wrapcauchy_ppf_analytical(q, x[1], loc=x[0])

    sample = circular_quantile_sampling(ppf_func, len(given_data_normed_sorted)) / (
        2 * np.pi
    )
    return circular_wasserstein_from_samples(
        given_data_normed_sorted, sample, p=1, sorted=True
    )


def W1_quantile_sampling(
    given_data: npt.NDArray[np.float64],
    x0: Optional[npt.NDArray[np.float64]] = None,
    method="powell",
) -> optimize.OptimizeResult:
    """1-Wasserstein 距離（分位点サンプリング）を最小化するパラメータ推定"""
    given_data = to_2pi_range(given_data)
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(
        W1_quantile_sampling_cost_func, given_data_normed_sorted=given_data_norm_sorted
    )
    if method == "differential_evolution":
        return optimize.differential_evolution(cost_func, bounds=bounds)
    else:
        if x0 is None:
            raise ValueError("x0 is required for local minimization")
        return optimize.minimize(
            cost_func,
            x0,
            bounds=bounds,
            method=method,
            options={"xtol": 1e-6, "ftol": 1e-6},
        )


def W2_quantile_sampling_cost_func(
    x, given_data_normed_sorted: npt.NDArray[np.float64]
) -> float:
    def ppf_func(q):
        return wrapcauchy_ppf_analytical(q, x[1], loc=x[0])

    sample = circular_quantile_sampling(ppf_func, len(given_data_normed_sorted)) / (
        2 * np.pi
    )
    return circular_wasserstein_from_samples(
        given_data_normed_sorted, sample, p=2, sorted=True
    )


def W2_quantile_sampling(
    given_data: npt.NDArray[np.float64],
    x0: Optional[npt.NDArray[np.float64]] = None,
    method="powell",
) -> optimize.OptimizeResult:
    """2-Wasserstein 距離（分位点サンプリング）を最小化するパラメータ推定"""
    given_data = to_2pi_range(given_data)
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(
        W2_quantile_sampling_cost_func, given_data_normed_sorted=given_data_norm_sorted
    )
    if method == "differential_evolution":
        return optimize.differential_evolution(cost_func, bounds=bounds)
    else:
        if x0 is None:
            raise ValueError("x0 is required for local minimization")
        return optimize.minimize(
            cost_func,
            x0,
            bounds=bounds,
            method=method,
            options={"xtol": 1e-6, "ftol": 1e-6},
        )


def circular_variance(rho: float) -> float:
    """円周分散"""
    return 1 - rho
