"""
実験5
巻き込みコーシー分布のパラメータとワッサースタイン距離のグラフを描画して、関数の形を調べる

結果: rho=0.1という一様分布に近い場合に勾配が緩くなり、ネルダーミード法がいい結果をださない。
一方でパウエル法はいい結果を返す。
"""

import time

import numpy as np
import scipy.stats as stats
from scipy import optimize

from ..calc_semidiscrete_W_dist import method1, method2
from ..cauchy import MLE_wrapped_cauchy_OKAMURA_method, wrapped_cauchy_cumsum_hist
from ..plots import brute_heatmap


def est_W1_method2(given_data):
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[0, 2*pi]のデータ

    Returns:
        Tuple[float, float]: [推定したmu、推定したrho]
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist_data(
        given_data, len(given_data)
    )

    def cost_func(x):
        mu, rho = x
        dist_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist(mu, rho, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.brute(
        cost_func,
        ((0, 2 * np.pi), (0.01, 0.99)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def est_W1_method2_justopt(given_data):
    bin_num = len(given_data)
    data_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist_data(
        given_data, len(given_data)
    )

    def cost_func(x):
        mu, rho = x
        dist_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist(mu, rho, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=((0, 2 * np.pi), (0.01, 0.99)),
        # for powell method
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
        # for Nelder-Mead method
        # method="Nelder-Mead",
        # options={"xatol": 1e-8, "fatol": 1e-8},
    )


def est_W2_method1(given_data):
    """Calc W2-estimator using method1"""
    given_data_norm = np.remainder(given_data, 2 * np.pi) / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)

    def cost_func(x):
        mu, rho = x
        sample = stats.wrapcauchy(loc=mu, c=rho).rvs(len(given_data))
        sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
        sample = np.sort(sample)
        return method1.method1(given_data_norm_sorted, sample, p=2, sorted=True)

    return optimize.brute(
        cost_func,
        ((0, 2 * np.pi), (0.01, 0.99)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def est_W2_method1_justopt(given_data):
    given_data_norm = np.remainder(given_data, 2 * np.pi) / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)

    def cost_func(x):
        mu, rho = x
        sample = stats.wrapcauchy(loc=mu, c=rho).rvs(len(given_data))
        sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
        sample = np.sort(sample)
        return method1.method1(given_data_norm_sorted, sample, p=2, sorted=True)

    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=((0, 2 * np.pi), (0.01, 0.99)),
        # for powell method
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
        # for Nelder-Mead method
        # method="Nelder-Mead",
        # options={"xatol": 1e-8, "fatol": 1e-8},
    )


def main():
    N = 1000
    mu = np.pi / 2
    rho = 0.8

    print(f"N={N}, True parameter: mu={mu}, rho={rho}")

    # [0, 2*pi] の範囲でサンプリングしたいが、[mu, mu + 2*pi] の範囲になっているので修正
    sample = stats.wrapcauchy(loc=mu, c=rho).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    s_time = time.perf_counter()
    MLE = MLE_wrapped_cauchy_OKAMURA_method.calc_MLE(sample, N, iter_num=100)
    e_time = time.perf_counter()
    print(f"MLE result: mu={np.angle(MLE)}, rho={np.abs(MLE)}, time={e_time-s_time}s")

    s_time = time.perf_counter()
    ret = est_W1_method2(sample)
    e_time = time.perf_counter()
    print(f"Mehtod2 Estimation result: time={e_time-s_time}s")
    brute_heatmap.plot_heatmap(ret, ("mu", "rho"))

    s_time = time.perf_counter()
    ret2 = est_W1_method2_justopt(sample)
    e_time = time.perf_counter()
    print(f"Mehtod2 Estimation result with just optimization: time={e_time-s_time}s")
    print(ret2)

    s_time = time.perf_counter()
    ret3 = est_W2_method1(sample)
    e_time = time.perf_counter()
    print(f"Mehtod1 Estimation result: time={e_time-s_time}s")
    brute_heatmap.plot_heatmap(ret3, ("mu", "rho"))

    s_time = time.perf_counter()
    ret4 = est_W2_method1_justopt(sample)
    e_time = time.perf_counter()
    print(f"Mehtod1 Estimation result with just optimization: time={e_time-s_time}s")
    print(ret4)


if __name__ == "__main__":
    main()
