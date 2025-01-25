"""
何回かサンプルをとってMSEを計算する
MSE, W2-estimator(method1), W1-estimator(method2)の比較
"""

import time
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import optimize

from ..calc_semidiscrete_W_dist import method1, method2
from ..vonmises import vonmises_cumsum_hist
from ..vonmises import vonmises_MLE as vonmises_MLE


def est_method1(given_data) -> Tuple[float, float]:
    """Calc W2-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)

    def cost_func(x):
        sample = stats.vonmises(loc=x[0], kappa=x[1]).rvs(len(given_data))
        sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
        return method1.method1(given_data_norm, sample, p=2)

    bounds = ((-np.pi, np.pi), (0, 10))
    finish_func = partial(optimize.minimize, method="powell", bounds=bounds)

    return optimize.brute(
        cost_func,
        bounds,
        full_output=True,
        finish=finish_func,
        Ns=100,
    )


def est_method2(given_data) -> Tuple[float, float]:
    """Calc W1-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = vonmises_cumsum_hist.cumsum_hist_data(given_data, bin_num)

    def cost_func(x):
        mu, kappa = x
        dist_cumsum_hist = vonmises_cumsum_hist.cumsum_hist(mu, kappa, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.minimize(
        cost_func,
        (0, 1),
        bounds=((-np.pi, np.pi), (0, 10)),
        # for powell method
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def main():
    true_mu = 0.3
    true_kappa = 2
    # Ns = [100, 500, 1000, 5000, 10000]
    Ns = [1000]
    # try_nums = [100, 100, 100, 100, 100]
    try_nums = [10]
    for N, try_num in zip(Ns, try_nums, strict=True):  # データ数Nを変える
        print(f"N={N}")
        MLE_mu = np.zeros(try_num)
        MLE_kappa = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        method1_mu = np.zeros(try_num)
        method1_kappa = np.zeros(try_num)
        method1_time = np.zeros(try_num)
        method2_mu = np.zeros(try_num)
        method2_kappa = np.zeros(try_num)
        method2_time = np.zeros(try_num)

        for i in range(try_num):  # MSEをとるための試行回数
            sample = stats.vonmises(loc=true_mu, kappa=true_kappa).rvs(N)
            sample = np.remainder(sample, 2 * np.pi)

            s_time = time.perf_counter()
            MLE = vonmises_MLE.MLE(vonmises_MLE.T(sample), N)
            e_time = time.perf_counter()
            MLE_mu[i] = MLE[0]
            MLE_kappa[i] = MLE[1]
            MLE_time[i] = e_time - s_time

            s_time = time.perf_counter()
            # est = est_method1(sample)
            e_time = time.perf_counter()
            # method1_mu[i] = est[0][0]
            # method1_kappa[i] = est[0][1]
            method1_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = est_method2(sample)
            e_time = time.perf_counter()
            method2_mu[i] = est.x[0]
            method2_kappa[i] = est.x[1]
            method2_time[i] = e_time - s_time

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_kappa_mse = np.mean((MLE_kappa - true_kappa) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        method1_mu_mse = np.mean((method1_mu - true_mu) ** 2)
        method1_kappa_mse = np.mean((method1_kappa - true_kappa) ** 2)
        method1_time_mean = np.mean(method1_time)
        method2_mu_mse = np.mean((method2_mu - true_mu) ** 2)
        method2_kappa_mse = np.mean((method2_kappa - true_kappa) ** 2)
        method2_time_mean = np.mean(method2_time)

        print(
            f"MLE: mu_mse={MLE_mu_mse}, kappa_mse={MLE_kappa_mse}, time={MLE_time_mean}"
        )
        # print(
        #     f"method1: mu_mse={method1_mu_mse}, kappa_mse={method1_kappa_mse}, time={method1_time_mean}"
        # )
        print(
            f"method2: mu_mse={method2_mu_mse}, kappa_mse={method2_kappa_mse}, time={method2_time_mean}"
        )
        print(method2_mu)
        print(method2_kappa)


if __name__ == "__main__":
    main()
