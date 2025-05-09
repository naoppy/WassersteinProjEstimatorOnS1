"""
何回かサンプルをとってMSEを計算する
MSE, W1-estimator(method2), W2-estimator(method3)の比較
"""

import multiprocessing
import time
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import ray
import scipy.stats as stats
from numpy import typing as npt
from scipy import optimize
from scipy.stats import vonmises as vonmises_scipy
from tqdm import tqdm

from ..calc_semidiscrete_W_dist import method1, method2
from ..distributions import vonmises

bounds = ((-np.pi, np.pi), (0.1, 5))


def W2_cost_func3(x, given_data_normed_sorted):
    sample = vonmises.fast_quantile_sampling(x[0], x[1], len(given_data_normed_sorted))
    sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
    sample = np.sort(sample)
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def est_W2_method3(given_data):
    """Calc W2-estimator using method3

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W2_cost_func3, given_data_normed_sorted=given_data_norm_sorted)

    # return optimize.differential_evolution(
    #     cost_func, tol=0.01, bounds=bounds, workers=-1, updating="deferred"
    # )
    return optimize.minimize(
        cost_func,
        (0, 2.5),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def W2_cost_func1(x, given_data_normed_sorted):
    sample = vonmises_scipy(loc=x[0], kappa=x[1]).rvs(len(given_data_normed_sorted))
    sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
    sample = np.sort(sample)
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def est_W2_method1(given_data):
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W2_cost_func1, given_data_normed_sorted=given_data_norm_sorted)
    return optimize.differential_evolution(
        cost_func, tol=0.01, bounds=bounds, workers=-1, updating="deferred"
    )


def W1_cost_func(x, bin_num, data_cumsum_hist):
    mu, kappa = x
    dist_cumsum_hist = vonmises.cumsum_hist(mu, kappa, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def est_W1_method2(given_data):
    """Calc W1-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = vonmises.cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        W1_cost_func, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    # return optimize.differential_evolution(
    #     cost_func, tol=0.01, bounds=bounds, workers=-1, updating="deferred"
    # )
    return optimize.minimize(
        cost_func,
        (0, 2.5),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


@ray.remote
def run_once(true_mu, true_kappa, N: int) -> npt.NDArray[np.float64]:
    sample = stats.vonmises(loc=true_mu, kappa=true_kappa).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    s_time = time.perf_counter()
    MLE = vonmises.MLE(vonmises.T(sample), N)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_kappa = MLE[1]
    MLE_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_W1_method2(sample)
    e_time = time.perf_counter()
    method2_mu = est.x[0]
    method2_kappa = est.x[1]
    method2_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_W2_method3(sample)
    e_time = time.perf_counter()
    method3_mu = est.x[0]
    method3_kappa = est.x[1]
    method3_time = e_time - s_time

    return np.array([MLE_mu, MLE_kappa, MLE_time,
            method2_mu, method2_kappa, method2_time,
            method3_mu, method3_kappa, method3_time])


def main():
    ray.init(num_cpus=multiprocessing.cpu_count())
    # 実験条件1
    true_mu = 0.3
    true_kappa = 2
    # 実験条件2
    # true_mu = -np.pi / 2
    # true_kappa = 0.4
    print(f"true mu={true_mu}, true kappa={true_kappa}")
    print("(mu, kappa, time)")

    log10_Ns = [2, 2.5, 3, 3.5, 4, 4.5]
    Ns = np.power(10, log10_Ns).astype(np.int64)
    try_nums = [1000] * len(Ns)

    df = pd.DataFrame(
        index=log10_Ns,
        columns=[
            "MLE_mu",
            "MLE_kappa",
            "W1(method2)_mu",
            "W1(method2)_kappa",
            "W2(method3)_mu",
            "W2(method3)_kappa",
            "Cramer-Rao Lower Bound of mu",
            "Cramer-Rao Lower Bound of kappa",
        ],
    )
    fisher_mat_inv_diag = vonmises.fisher_mat_inv_diag(true_kappa)

    for i, (N, try_num) in enumerate(
        zip(Ns, try_nums, strict=True)
    ):  # データ数Nを変える
        print(f"N={N}")
        MLE_mu = np.zeros(try_num)
        MLE_kappa = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        method2_mu = np.zeros(try_num)
        method2_kappa = np.zeros(try_num)
        method2_time = np.zeros(try_num)
        method3_mu = np.zeros(try_num)
        method3_kappa = np.zeros(try_num)
        method3_time = np.zeros(try_num)

        ray_ids = []
        for _ in range(try_num):  # MSEをとるための試行回数
            ret = run_once.remote(true_mu, true_kappa, N)
            ray_ids.append(ret)
        result = ray.get(ray_ids) # (1000, 9)
        for i in range(try_num):
            r = result[i]
            MLE_mu[i] = r[0]
            MLE_kappa[i] = r[1]
            MLE_time[i] = r[2]
            method2_mu[i] = r[3]
            method2_kappa[i] = r[4]
            method2_time[i] = r[5]
            method3_mu[i] = r[6]
            method3_kappa[i] = r[7]
            method3_time[i] = r[8]

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_kappa_mse = np.mean((MLE_kappa - true_kappa) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        method2_mu_mse = np.mean((method2_mu - true_mu) ** 2)
        method2_kappa_mse = np.mean((method2_kappa - true_kappa) ** 2)
        method2_time_mean = np.mean(method2_time)
        method3_mu_mse = np.mean((method3_mu - true_mu) ** 2)
        method3_kappa_mse = np.mean((method3_kappa - true_kappa) ** 2)
        method3_time_mean = np.mean(method3_time)
        df.loc[log10_Ns[i]] = [
            np.log10(MLE_mu_mse),
            np.log10(MLE_kappa_mse),
            np.log10(method2_mu_mse),
            np.log10(method2_kappa_mse),
            np.log10(method3_mu_mse),
            np.log10(method3_kappa_mse),
            np.log10(fisher_mat_inv_diag[0]) - log10_Ns[i],
            np.log10(fisher_mat_inv_diag[1]) - log10_Ns[i],
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse}, kappa_mse={MLE_kappa_mse}, time={MLE_time_mean}"
        )
        print(
            f"W1 method2: mu_mse={method2_mu_mse}, kappa_mse={method2_kappa_mse}, time={method2_time_mean}"
        )
        print(
            f"W2 method3: mu_mse={method3_mu_mse}, kappa_mse={method3_kappa_mse}, time={method3_time_mean}"
        )

    print(df)
    df.to_csv("./data/ex6_vonmises_MSE.csv")


if __name__ == "__main__":
    main()
