"""
何回かサンプルをとってMSEを計算する
MSE, W2-estimator(method1), W1-estimator(method2)の比較
"""

import time
from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy import typing as npt
from parfor import pmap
from scipy import optimize

from ..calc_semidiscrete_W_dist import method1, method2
from ..distributions import wrapedcauchy

bounds = ((-np.pi, np.pi), (0.01, 0.99))


def W2_cost_func3(x, given_data_normed_sorted):
    sample = wrapedcauchy.quantile_sampling(
        x[0], x[1], len(given_data_normed_sorted)
    ) / (2 * np.pi)
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def est_W2_method3(given_data: npt.NDArray[np.float64]):
    """Calc W2-estimator using method3

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W2_cost_func3, given_data_normed_sorted=given_data_norm_sorted)
    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def W1_method2_cost_func(x, bin_num, data_cumsum_hist):
    mu, rho = x
    dist_cumsum_hist = wrapedcauchy.cumsum_hist(mu, rho, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def est_W1_method2(given_data):
    """Calc W1-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrapedcauchy.cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        W1_method2_cost_func, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def W1_cost_func3(x, given_data_normed_sorted):
    sample = wrapedcauchy.quantile_sampling(
        x[0], x[1], len(given_data_normed_sorted)
    ) / (2 * np.pi)
    return method1.method1(given_data_normed_sorted, sample, p=1, sorted=True)


def est_W1_method3(given_data):
    """Calc W1-estimator using method3

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W1_cost_func3, given_data_normed_sorted=given_data_norm_sorted)
    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def run_once(i, true_mu, true_rho, N: int) -> npt.NDArray[np.float64]:
    # [0, 2*pi] の範囲でサンプリングしたいが、[mu, mu + 2*pi] の範囲になっているので修正
    sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    # s_time = time.perf_counter()
    # MLE = wrapedcauchy.MLE_OKAMURA(sample, N, iter_num=10000)
    # e_time = time.perf_counter()
    # MLE_mu_okamura[i] = MLE[0]
    # MLE_rho_okamura[i] = MLE[1]
    # MLE_time_okamura[i] = e_time - s_time

    s_time = time.perf_counter()
    MLE = wrapedcauchy.MLE_Kent(sample, tol=1e-15)
    e_time = time.perf_counter()
    MLE_mu_kent = MLE[0]
    MLE_rho_kent = MLE[1]
    MLE_time_kent = e_time - s_time

    # s_time = time.perf_counter()
    # MLE = wrapedcauchy.MLE_direct_opt(sample)
    # e_time = time.perf_counter()
    # MLE_mu_direct[i] = MLE[0]
    # MLE_rho_direct[i] = MLE[1]
    # MLE_time_direct[i] = e_time - s_time

    # s_time = time.perf_counter()
    # est = est_W2_method1(sample)
    # e_time = time.perf_counter()
    # method1_mu[i] = est[0][0]
    # method1_rho[i] = est[0][1]
    # method1_time[i] = e_time - s_time

    s_time = time.perf_counter()
    est = est_W1_method2(sample)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_rho = est.x[1]
    W1method2_time = e_time - s_time

    s_time = time.perf_counter()
    # profiler = cProfile.Profile()
    # profiler.enable()
    est = est_W2_method3(sample)
    # profiler.disable()
    # profiler.print_stats(sort="time")
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_rho = est.x[1]
    W2method3_time = e_time - s_time

    # s_time = time.perf_counter()
    # est = est_W1_method3(sample)
    # e_time = time.perf_counter()
    # method4_mu[i] = est.x[0]
    # method4_rho[i] = est.x[1]
    # method4_time[i] = e_time - s_time

    return np.array(
        [
            MLE_mu_kent,
            MLE_rho_kent,
            MLE_time_kent,
            W1method2_mu,
            W1method2_rho,
            W1method2_time,
            W2method3_mu,
            W2method3_rho,
            W2method3_time,
        ]
    )


def _main():
    true_mu = np.pi / 8
    N = int(np.power(10, 5))
    print(f"true mu={true_mu}, N={N}")
    print("(mu, rho, time)")

    rhos = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    try_nums = [1000] * len(rhos)

    df = pd.DataFrame(
        index=rhos,
        columns=[
            "MLE_mu / CR_mu",
            "MLE_rho / CR_rho",
            "W1(method2)_mu / MLE_mu",
            "W1(method2)_rho / MLE_rho",
            "W2(method3)_mu / MLE_mu",
            "W2(method3)_rho / MLE_rho",
        ],
    )
    df.index.name = "rho"

    for j, (true_rho, try_num) in enumerate(
        zip(rhos, try_nums, strict=True)
    ):  # データ数Nを変える
        print(f"rho={true_rho}")
        MLE_mu_kent = np.zeros(try_num)
        MLE_rho_kent = np.zeros(try_num)
        MLE_time_kent = np.zeros(try_num)
        W1method2_mu = np.zeros(try_num)
        W1method2_rho = np.zeros(try_num)
        W1method2_time = np.zeros(try_num)
        W2method3_mu = np.zeros(try_num)
        W2method3_rho = np.zeros(try_num)
        W2method3_time = np.zeros(try_num)

        result = pmap(run_once, range(try_num), (true_mu, true_rho, N))
        for i in range(try_num):
            r = result[i]
            MLE_mu_kent[i] = r[0]
            MLE_rho_kent[i] = r[1]
            MLE_time_kent[i] = r[2]
            W1method2_mu[i] = r[3]
            W1method2_rho[i] = r[4]
            W1method2_time[i] = r[5]
            W2method3_mu[i] = r[6]
            W2method3_rho[i] = r[7]
            W2method3_time[i] = r[8]

        # MSEを計算する
        MLE_mu_kent_mse = np.mean((MLE_mu_kent - true_mu) ** 2)
        MLE_rho_kent_mse = np.mean((MLE_rho_kent - true_rho) ** 2)
        MLE_time_kent_mean = np.mean(MLE_time_kent)
        method2_mu_mse = np.mean((W1method2_mu - true_mu) ** 2)
        method2_rho_mse = np.mean((W1method2_rho - true_rho) ** 2)
        method2_time_mean = np.mean(W1method2_time)
        method3_mu_mse = np.mean((W2method3_mu - true_mu) ** 2)
        method3_rho_mse = np.mean((W2method3_rho - true_rho) ** 2)
        method3_time_mean = np.mean(W2method3_time)

        fisher_mat_inv_diag = wrapedcauchy.fisher_mat_inv_diag(true_rho)
        CR_mu_mse_times_N = fisher_mat_inv_diag[0]
        CR_rho_mse_times_N = fisher_mat_inv_diag[1]

        df.loc[true_rho] = [
            N * MLE_mu_kent_mse / CR_mu_mse_times_N,
            N * MLE_rho_kent_mse / CR_rho_mse_times_N,
            method2_mu_mse / MLE_mu_kent_mse,
            method2_rho_mse / MLE_rho_kent_mse,
            method3_mu_mse / MLE_mu_kent_mse,
            method3_rho_mse / MLE_rho_kent_mse,
        ]

        print(
            f"MLE kent: mu_mse={MLE_mu_kent_mse}, rho_mse={MLE_rho_kent_mse}, time={MLE_time_kent_mean}"
        )
        print(
            f"W1 method2: mu_mse={method2_mu_mse}, rho_mse={method2_rho_mse}, time={method2_time_mean}"
        )
        print(
            f"W2 method3: mu_mse={method3_mu_mse}, rho_mse={method3_rho_mse}, time={method3_time_mean}"
        )

    print(df)
    df.to_csv("./data/ex75_wrapcauchy_change_rho.csv")


if __name__ == "__main__":
    _main()
