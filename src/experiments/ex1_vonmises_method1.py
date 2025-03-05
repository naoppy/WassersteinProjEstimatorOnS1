"""
連続分布のパラメータを、経験分布関数との2-ワッサースタイン距離最小化で推定します。
このとき、連続分布からサンプリングした離散分布で、元の連続分布を近似します。
"""

import time
from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from ..calc_semidiscrete_W_dist.method1 import method1
from ..distributions import vonmises


def estimate_param(given_data) -> Tuple[float, float]:
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        Tuple[float, float]: [推定した平均、推定した分散]
    """
    given_data_norm = given_data / (2 * np.pi)

    def cost_func(x):
        sample = stats.vonmises(loc=x[0], kappa=x[1]).rvs(len(given_data)) / (2 * np.pi)
        return method1(given_data_norm, sample, p=2)

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        full_output=True,
        finish=optimize.fmin_powell,
        Ns=100,
    )[0]


def estimate_param2(given_data) -> Tuple[float, float]:
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        Tuple[float, float]: [推定した平均、推定した分散]
    """
    given_data_norm = given_data / (2 * np.pi)

    def cost_func(x):
        sample = np.remainder(
            vonmises.quantile_sampling(x[0], x[1], len(given_data)), 2 * np.pi
        ) / (2 * np.pi)
        return method1(given_data_norm, sample, p=2)

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        full_output=True,
        finish=optimize.fmin_powell,
        Ns=100,
    )[0]


def main():
    N = 1000
    mu1 = 0.3
    kappa1 = 2

    print(f"N={N}, True parameter: mu={mu1}, kappa={kappa1}")

    sample = stats.vonmises(loc=mu1, kappa=kappa1).rvs(N)
    # vonmisesMLE.plot_vonmises(sample, mu1, kappa1, N)

    time1 = time.perf_counter()
    T_data = vonmises.T(sample)
    mu_MLE, kappa_MLE = vonmises.MLE(T_data, N)
    time2 = time.perf_counter()
    print(f"MLE result: mu={mu_MLE}, kappa={kappa_MLE}, time={time2-time1}s")

    time3 = time.perf_counter()
    mu_est, kappa_est = estimate_param(sample)
    time4 = time.perf_counter()
    print(f"Estimation result: mu={mu_est}, kappa={kappa_est}, time={time4-time3}s")

    time5 = time.perf_counter()
    mu_est2, kappa_est2 = estimate_param2(sample)
    time6 = time.perf_counter()
    print(f"Estimation result2: mu={mu_est2}, kappa={kappa_est2}, time={time6-time5}s")


if __name__ == "__main__":
    main()
