"""
何回かサンプルをとってMSEを計算する
MSE, W1-estimator(method2), W2-estimator(method3)の比較
"""

import time
from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy import typing as npt
from parfor import pmap
from scipy import optimize
from scipy.stats import vonmises as vonmises_scipy

from ..calc_semidiscrete_W_dist import method1, method2
from ..distributions import vonmises

bounds = ((-np.pi, np.pi), (0.1, 7.0))
initial_guess = (0, 3.5)


def Wp_cost_func3(x, given_data_normed_sorted, p: int):
    sample = vonmises.fast_quantile_sampling(x[0], x[1], len(given_data_normed_sorted))
    sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
    sample = np.sort(sample)
    return method1.method1(given_data_normed_sorted, sample, p=p, sorted=True)


def est_W1_method3(given_data):
    """calc W1E by method3, given_data should be in [0, 2pi]"""
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(
        Wp_cost_func3, given_data_normed_sorted=given_data_norm_sorted, p=1
    )
    return optimize.minimize(
        cost_func,
        initial_guess,
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def est_W2_method3(given_data):
    """Calc W2-estimator using method3

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(
        Wp_cost_func3, given_data_normed_sorted=given_data_norm_sorted, p=2
    )
    return optimize.minimize(
        cost_func,
        initial_guess,
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
    """method1: random sampling, not used now"""
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W2_cost_func1, given_data_normed_sorted=given_data_norm_sorted)
    return optimize.differential_evolution(
        cost_func, tol=0.01, bounds=bounds, workers=-1, updating="deferred"
    )


def W1_cost_func2(x, bin_num, data_cumsum_hist):
    mu, kappa = x
    dist_cumsum_hist = vonmises.cumsum_hist(mu, kappa, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def est_W1_method2(given_data):
    """Calc W1-estimator using method2

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = vonmises.cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        W1_cost_func2, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    return optimize.minimize(
        cost_func,
        initial_guess,
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def run_once(i, true_mu, true_kappa, N: int) -> npt.NDArray[np.float64]:
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
    W1method2_mu = est.x[0]
    W1method2_kappa = est.x[1]
    W1method2_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_W2_method3(sample)
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_kappa = est.x[1]
    W2method3_time = e_time - s_time

    return np.array(
        [
            MLE_mu,
            MLE_kappa,
            MLE_time,
            W1method2_mu,
            W1method2_kappa,
            W1method2_time,
            W2method3_mu,
            W2method3_kappa,
            W2method3_time,
        ]
    )


def main():
    # 実験条件1
    true_mu = 0.3
    kappas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    try_nums = [1000] * len(kappas)
    N = int(np.power(10, 4))
    print("N=", N, "true_mu=", true_mu)
    print("(mu, kappa, time)")

    df = pd.DataFrame(
        index=kappas,
        columns=[
            "MLE_mu / CR_mu",
            "MLE_kappa / CR_kappa",
            "W1(method2)_mu / CR_mu",
            "W1(method2)_kappa / CR_kappa",
            "W2(method3)_mu / CR_mu",
            "W2(method3)_kappa / CR_kappa",
        ],
    )

    for j, (true_kappa, try_num) in enumerate(
        zip(kappas, try_nums, strict=True)
    ):  # データ数Nを変える
        print(f"kappa={true_kappa}")
        MLE_mu = np.zeros(try_num)
        MLE_kappa = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        W1method2_mu = np.zeros(try_num)
        W1method2_kappa = np.zeros(try_num)
        W1method2_time = np.zeros(try_num)
        W2method3_mu = np.zeros(try_num)
        W2method3_kappa = np.zeros(try_num)
        W2method3_time = np.zeros(try_num)

        # MSEをとるための試行回数
        result = pmap(run_once, range(try_num), (true_mu, true_kappa, N))
        for i in range(try_num):
            r = result[i]
            MLE_mu[i] = r[0]
            MLE_kappa[i] = r[1]
            MLE_time[i] = r[2]
            W1method2_mu[i] = r[3]
            W1method2_kappa[i] = r[4]
            W1method2_time[i] = r[5]
            W2method3_mu[i] = r[6]
            W2method3_kappa[i] = r[7]
            W2method3_time[i] = r[8]

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_kappa_mse = np.mean((MLE_kappa - true_kappa) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        W1method2_mu_mse = np.mean((W1method2_mu - true_mu) ** 2)
        W1method2_kappa_mse = np.mean((W1method2_kappa - true_kappa) ** 2)
        W1method2_time_mean = np.mean(W1method2_time)
        W2method3_mu_mse = np.mean((W2method3_mu - true_mu) ** 2)
        W2method3_kappa_mse = np.mean((W2method3_kappa - true_kappa) ** 2)
        W2method3_time_mean = np.mean(W2method3_time)
        CR_mu = vonmises.fisher_mat_inv_diag(true_kappa)[0] / N
        CR_kappa = vonmises.fisher_mat_inv_diag(true_kappa)[1] / N
        df.loc[true_kappa] = [
            MLE_mu_mse / CR_mu,
            MLE_kappa_mse / CR_kappa,
            W1method2_mu_mse / CR_mu,
            W1method2_kappa_mse / CR_kappa,
            W2method3_mu_mse / CR_mu,
            W2method3_kappa_mse / CR_kappa,
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse}, kappa_mse={MLE_kappa_mse}, time={MLE_time_mean}"
        )
        print(
            f"W1 method2: mu_mse={W1method2_mu_mse}, kappa_mse={W1method2_kappa_mse}, time={W1method2_time_mean}"
        )
        print(
            f"W2 method3: mu_mse={W2method3_mu_mse}, kappa_mse={W2method3_kappa_mse}, time={W2method3_time_mean}"
        )

    print(df)
    df.to_csv("./data/ex65_MSE_kappa_change.csv")


if __name__ == "__main__":
    main()
