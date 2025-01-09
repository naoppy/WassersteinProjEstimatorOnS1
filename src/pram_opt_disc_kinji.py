"""
連続分布のパラメータを、経験分布関数との2-ワッサースタイン距離最小化で推定します。
このとき、連続分布からサンプリングした離散分布で、元の連続分布を近似します。
"""

from datetime import datetime
import time
from typing import Any, Dict, Tuple
import matplotlib.pylab as plt
import numpy as np
import ot
from scipy import optimize
from scipy.stats import vonmises
import vonmises_MLE
from scipy.optimize import minimize


def param_cost(est_mu, est_kappa, given_data) -> float:
    """推定したパラメータからサンプリングした分布とデータとのWasserstein距離を計算する

    Args:
        est_mu (float): 推定した平均
        est_kappa (float): 推定した分散
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        float: W2距離
    """
    n = len(given_data)
    given_data_norm1 = given_data / (2 * np.pi)
    x2_norm1 = vonmises(loc=est_mu, kappa=est_kappa).rvs(n) / (2 * np.pi)
    return ot.binary_search_circle(given_data_norm1, x2_norm1, p=2, log=False)


def estimate_param(given_data) -> Tuple[float, float]:
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        Tuple[float, float]: [推定した平均、推定した分散]
    """

    def cost_func(x):
        return param_cost(x[0], x[1], given_data)

    # res = minimize(
    #     cost_func, [0, 1], method="BFGS", bounds=[(-np.pi, np.pi), (0, np.inf)]
    # )
    # return res.x
    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        full_output=True,
        finish=optimize.fmin_powell,
        Ns=100,
    )[0]


def main():
    N = 500
    mu1 = 0.3
    kappa1 = 2
    sample = vonmises(loc=mu1, kappa=kappa1).rvs(N)
    vonmises_MLE.plot_vonmises(sample, mu1, kappa1, N)
    time1 = time.perf_counter()
    T_data = vonmises_MLE.T(sample)
    mu_MLE, kappa_MLE = vonmises_MLE.MLE(T_data, N)
    time2 = time.perf_counter()
    print(f"MLE result: mu={mu_MLE}, kappa={kappa_MLE}, time={time2-time1}s")
    time3 = time.perf_counter()
    mu_est, kappa_est = estimate_param(sample)
    time4 = time.perf_counter()
    print(f"Estimation result: mu={mu_est}, kappa={kappa_est}, time={time4-time3}s")


if __name__ == "__main__":
    main()
