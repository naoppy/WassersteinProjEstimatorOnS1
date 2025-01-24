"""
実験3
巻き込みコーシー分布について、MLE, method1, method2を比較する
"""

import time
from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from ..calc_semidiscrete_W_dist import method1, method2
from ..cauchy import MLE_wrapped_cauchy_OKAMURA_method, wrapped_cauchy_cumsum_hist


def estimate_param(given_data) -> Tuple[float, float]:
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        Tuple[float, float]: [推定したmu、推定したrho]
    """
    n = len(given_data)
    bin_num = len(given_data)
    data_hist = np.zeros(bin_num + 1)
    for x in given_data:
        data_hist[np.clip(int(x / (2 * np.pi) * bin_num)+1, 1, bin_num)] += 1
    data_cumsum_hist = np.cumsum(data_hist) / n
    assert abs(data_cumsum_hist[0]-0.0) < 1e-7
    assert abs(data_cumsum_hist[-1]-1.0) < 1e-7

    def cost_func(x):
        mu, kappa = x
        dist_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist(mu, kappa, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        full_output=True,
        finish=optimize.fmin_powell,
        Ns=100,
    )[0]


def main():
    N = 10000
    mu = np.pi/2
    rho = 0.7

    print(f"N={N}, True parameter: mu={mu}, rho={rho}")

    sample = stats.wrapcauchy(loc=mu, c=rho).rvs(N)

    time1 = time.perf_counter()
    MLE = MLE_wrapped_cauchy_OKAMURA_method.calc_MLE(sample, N, iter_num=100)
    time2 = time.perf_counter()
    print(f"MLE result: mu={np.angle(MLE)}, rho={np.abs(MLE)}, time={time2-time1}s")

    time3 = time.perf_counter()
    mu_est, rho_est = estimate_param(sample)
    time4 = time.perf_counter()
    print(f"Mehtod2 Estimation result: mu={mu_est}, rho={rho_est}, time={time4-time3}s")


if __name__ == "__main__":
    main()
