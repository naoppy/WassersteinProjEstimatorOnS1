from functools import partial
from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.special import i0, i1, iv, ive
from scipy.stats import vonmises

from ..calc_semidiscrete_W_dist import method1, method2
from ..misc.circular_utils import (
    circular_quantile_sampling,
    cumsum_hist_data,
    to_2pi_range,
)

bounds = ((-np.pi, np.pi), (0.01, 100.0))


def _bessel_ratio(v: int, kappa: float) -> float:
    """I_v(kappa) / I_0(kappa) を安全に計算する。"""
    if kappa < 600:
        if v == 0:
            return 1.0
        elif v == 1:
            return i1(kappa) / i0(kappa)
        else:
            return iv(v, kappa) / i0(kappa)
    else:
        return ive(v, kappa) / ive(0, kappa)


def _bessel_ratio_i0(kappa1: float, kappa0: float) -> float:
    """I_0(kappa1) / I_0(kappa0) を安全に計算する。"""
    if max(kappa1, kappa0) < 600:
        return i0(kappa1) / i0(kappa0)
    else:
        return (ive(0, kappa1) / ive(0, kappa0)) * np.exp(kappa1 - kappa0)


def fisher_info_2x2(kappa: float) -> npt.NDArray[np.float64]:
    """フォンミーゼス分布のフィッシャー情報量を計算する"""
    r1 = _bessel_ratio(1, kappa)
    r2 = _bessel_ratio(2, kappa)
    return np.array(
        [
            [kappa * r1, 0],
            [0, (1 + r2) / 2 - r1**2],
        ]
    )


def fisher_mat_inv_diag(kappa: float) -> List[float]:
    """フィッシャー情報行列の逆行列の対角成分のリストを返す。"""
    mat = fisher_info_2x2(kappa)
    return [1 / mat[0][0], 1 / mat[1][1]]


def T(x: npt.NDArray[np.float64]) -> List[float]:
    """フォンミーゼス分布の十分統計量を返す"""
    return [np.sum(np.cos(x)), np.sum(np.sin(x))]


def MLE(T_data, N: int) -> List[float]:
    """十分統計量を用いた最尤推定"""
    mu_MLE = np.arctan2(T_data[1], T_data[0])
    target_value = (T_data[0] * np.cos(mu_MLE) + T_data[1] * np.sin(mu_MLE)) / N
    EPS = 1e-6
    left = EPS
    right = 100000.0
    while right - left > EPS:
        mid = (left + right) / 2
        now_value = _bessel_ratio(1, mid)
        if np.abs(now_value - target_value) < EPS:
            break
        elif now_value - target_value > 0:
            right = mid
        else:
            left = mid
    kappa_MLE = mid
    return [mu_MLE, kappa_MLE]


def vonmises_pdf_stable(
    theta: npt.NDArray[np.float64], mu: float, kappa: float
) -> npt.NDArray[np.float64]:
    """Stable von Mises PDF using ive to prevent overflow."""
    return np.exp(kappa * (np.cos(theta - mu) - 1)) / (2 * np.pi * ive(0, kappa))


def vonmises_periodic_cdf_numerical(
    x: npt.NDArray[np.float64], mu: float, kappa: float
) -> npt.NDArray[np.float64]:
    """Numerical periodic CDF for von Mises distribution.

    Normalized to start at 0 on [0, 2*pi].
    """
    dist = vonmises(loc=mu, kappa=kappa)
    # Scipy's vonmises.cdf is monotonic on the real line: cdf(x + 2*pi) = cdf(x) + 1.
    # Therefore, to normalize it to start at 0 at x=0, we simply compute:
    return dist.cdf(x) - dist.cdf(0)


def cumsum_hist(mu: float, kappa: float, bin_num: int) -> npt.NDArray[np.float64]:
    """[0, 2pi] を bin_num 等分した区間でのcdfの値を返す"""
    x = np.linspace(0, 2 * np.pi, bin_num + 1)
    y = vonmises_periodic_cdf_numerical(x, mu, kappa)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def quantile_sampling(
    mu: float, kappa: float, sample_num: int
) -> npt.NDArray[np.float64]:
    """von Mises 分布から分位点サンプリングする"""
    dist = vonmises(loc=mu, kappa=kappa)

    def ppf_func(q):
        return dist.ppf(q)

    return circular_quantile_sampling(ppf_func, sample_num)


def circular_variance(kappa: float) -> float:
    """円周分散"""
    R = _bessel_ratio(1, kappa)
    return 1 - R


def A0(kappa: float) -> float:
    """A0関数を計算する"""
    return _bessel_ratio(1, kappa)


def A0Inverse(y: float) -> float:
    """A0の逆関数を数値的に求める"""
    EPS = 1e-6
    left = EPS
    right = 100000.0
    while right - left > EPS:
        mid = (left + right) / 2
        now_value = _bessel_ratio(1, mid)
        if np.abs(now_value - y) < EPS:
            break
        elif now_value - y > 0:
            right = mid
        else:
            left = mid
    return mid


def MLE_direct(sample: npt.NDArray[np.float64]) -> List[float]:
    """十分統計量を用いた標準的な最尤推定"""
    sample = to_2pi_range(sample)
    T_data = T(sample)
    return MLE(T_data, len(sample))


def W1_equal_div_cost_func(
    x, bin_num: int, data_cumsum_hist: npt.NDArray[np.float64]
) -> float:
    mu, kappa = x
    dist_cumsum_hist = cumsum_hist(mu, kappa, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def W1_equal_div(
    given_data: npt.NDArray[np.float64], x0=None, method="powell"
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
            x0 = (0, 1.0)
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
    dist = vonmises(loc=x[0], kappa=x[1])

    def ppf_func(q):
        return dist.ppf(q)

    sample = circular_quantile_sampling(ppf_func, len(given_data_normed_sorted)) / (
        2 * np.pi
    )
    return method1.method1(given_data_normed_sorted, sample, p=1, sorted=True)


def W1_quantile_sampling(
    given_data: npt.NDArray[np.float64], x0=None, method="powell"
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
            x0 = (0, 1.0)
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
    dist = vonmises(loc=x[0], kappa=x[1])

    def ppf_func(q):
        return dist.ppf(q)

    sample = circular_quantile_sampling(ppf_func, len(given_data_normed_sorted)) / (
        2 * np.pi
    )
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def W2_quantile_sampling(
    given_data: npt.NDArray[np.float64], x0=None, method="powell"
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
            x0 = (0, 1.0)
        return optimize.minimize(
            cost_func,
            x0,
            bounds=bounds,
            method=method,
            options={"xtol": 1e-6, "ftol": 1e-6},
        )


def type0_estimate(
    data: npt.NDArray[np.float64], gamma: float, debug: bool = False
) -> List[float]:
    """Kato and Eguchi (2016) の type0 推定量"""
    data = to_2pi_range(data)
    T_data = T(data)
    N = len(data)
    initial_guess = MLE(T_data, N)

    now_mu: float = initial_guess[0]
    now_kappa: float = initial_guess[1]

    for i in range(1000):
        next_mu = now_mu
        next_kappa = now_kappa

        exponents = gamma * now_kappa * np.cos(data - now_mu)
        max_exp = np.max(exponents)
        w = np.exp(exponents - max_exp)
        w_sum = np.sum(w)
        y = np.sum(w * np.sin(data))
        x = np.sum(w * np.cos(data))
        next_mu = np.arctan2(y, x)
        target = np.hypot(x, y) / w_sum
        next_kappa = A0Inverse(target) / (1 + gamma)

        if (next_mu - now_mu) ** 2 + (next_kappa - now_kappa) ** 2 < 1e-16:
            break
        now_mu = next_mu
        now_kappa = next_kappa
        if debug:
            print(f"debug: i={i}, mu={now_mu}, kappa={now_kappa}")
    return [now_mu, now_kappa]


def type1_estimate(
    data: npt.NDArray[np.float64], beta: float, debug: bool = False
) -> List[float]:
    """Kato and Eguchi (2016) の type1 推定量"""
    data = to_2pi_range(data)
    T_data = T(data)
    N = len(data)
    initial_guess = MLE(T_data, N)

    now_mu: float = initial_guess[0]
    now_kappa: float = initial_guess[1]

    for i in range(1000):
        next_mu = now_mu
        next_kappa = now_kappa

        exponents = beta * now_kappa * np.cos(data - now_mu)
        max_exp = np.max(exponents)
        w = np.exp(exponents - max_exp)
        w_sum = np.sum(w)

        w_norm = w / w_sum
        y_norm = np.sum(w_norm * np.sin(data))
        x_norm = np.sum(w_norm * np.cos(data))
        next_mu = np.arctan2(y_norm, x_norm)

        r_i0_base = ive(0, (1 + beta) * now_kappa) / ive(0, now_kappa)
        D_base = r_i0_base * (A0((1 + beta) * now_kappa) - A0(now_kappa)) / now_kappa

        exp_diff = np.minimum(700.0, beta * now_kappa - max_exp)
        coeff = (N * D_base / w_sum) * np.exp(exp_diff)

        target = np.hypot(
            x_norm - coeff * np.cos(now_mu), y_norm - coeff * np.sin(now_mu)
        )
        next_kappa = A0Inverse(target)

        if (next_mu - now_mu) ** 2 + (next_kappa - now_kappa) ** 2 < 1e-16:
            break
        now_mu = next_mu
        now_kappa = next_kappa
        if debug:
            print(f"debug: i={i}, mu={now_mu}, kappa={now_kappa}")
    return [now_mu, now_kappa]


def _plot_for_slide():
    """スライド用の分布描画"""
    n = 100000
    mu = 0
    kappa = 1
    plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")
    x = np.linspace(-np.pi, np.pi, 1000)
    pdf_vals = vonmises_pdf_stable(x, mu, kappa)

    dist = vonmises(loc=mu, kappa=kappa)

    def ppf_func(q):
        return dist.ppf(q)

    sample = circular_quantile_sampling(ppf_func, n)
    sample = np.remainder(sample + np.pi, 2 * np.pi) - np.pi
    ticks = [0, 0.15, 0.3]

    left.plot(x, pdf_vals)
    left.set_yticks(ticks)
    number_of_bins = int(np.sqrt(n))
    left.hist(sample, density=True, bins=number_of_bins)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x, pdf_vals, label="PDF")
    right.set_yticks(ticks)
    right.hist(sample, density=True, bins=number_of_bins, label="Histogram")
    right.set_title("Polar plot")

    right.legend(bbox_to_anchor=(0.15, 1.06))
    plt.show()
