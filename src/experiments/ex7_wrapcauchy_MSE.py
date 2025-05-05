"""
何回かサンプルをとってMSEを計算する
MSE, W2-estimator(method1), W1-estimator(method2)の比較
"""

import time
from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
from tqdm import tqdm

from ..calc_semidiscrete_W_dist import method1, method2
from ..distributions import wrapedcauchy

bounds = ((-np.pi, np.pi), (0.01, 0.99))


def W2_cost_func3(x, given_data_normed_sorted):
    sample = wrapedcauchy.quantile_sampling(
        x[0], x[1], len(given_data_normed_sorted)
    ) / (2 * np.pi)
    # sample = np.sort(sample) already sorted in quantile_sampling
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def est_W2_method3(given_data):
    """Calc W2-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W2_cost_func3, given_data_normed_sorted=given_data_norm_sorted)
    # return optimize.minimize(
    #     cost_func,
    #     (0, 0.5),
    #     bounds=bounds,
    #     method="powell",
    #     options={"xtol": 1e-6, "ftol": 1e-6},
    # )
    return optimize.differential_evolution(
        cost_func, tol=0.01, bounds=bounds, workers=-1, updating="deferred"
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
    return optimize.differential_evolution(
        cost_func, tol=0.01, bounds=bounds, workers=-1, updating="deferred"
    )


def W1_cost_func3(x, given_data_normed_sorted):
    sample = wrapedcauchy.quantile_sampling(
        x[0], x[1], len(given_data_normed_sorted)
    ) / (2 * np.pi)
    # sample = np.sort(sample)
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


def main():
    true_mu = np.pi / 8
    # 実験条件1
    true_rho = 0.7
    # 実験条件2
    # true_rho = 0.2
    print(f"true mu={true_mu}, true rho={true_rho}")
    print("(mu, rho, time)")

    log10_Ns = [2, 2.5, 3, 3.5, 4, 4.5, 5]
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
            "Cramer-Rao Lower Bound of rho",
        ],
    )
    fisher_mat_inv_diag = wrapedcauchy.fisher_mat_inv_diag(true_rho)

    for j, (N, try_num) in zip(Ns, try_nums, strict=True):  # データ数Nを変える
        print(f"N={N}")
        # MLE_mu_okamura = np.zeros(try_num)
        # MLE_rho_okamura = np.zeros(try_num)
        # MLE_time_okamura = np.zeros(try_num)
        MLE_mu_kent = np.zeros(try_num)
        MLE_rho_kent = np.zeros(try_num)
        MLE_time_kent = np.zeros(try_num)
        # MLE_mu_direct = np.zeros(try_num)
        # MLE_rho_direct = np.zeros(try_num)
        # MLE_time_direct = np.zeros(try_num)
        # method1_mu = np.zeros(try_num)
        # method1_rho = np.zeros(try_num)
        # method1_time = np.zeros(try_num)
        method2_mu = np.zeros(try_num)
        method2_rho = np.zeros(try_num)
        method2_time = np.zeros(try_num)
        method3_mu = np.zeros(try_num)
        method3_rho = np.zeros(try_num)
        method3_time = np.zeros(try_num)
        # method4_mu = np.zeros(try_num)
        # method4_rho = np.zeros(try_num)
        # method4_time = np.zeros(try_num)

        for i in tqdm(range(try_num)):  # MSEをとるための試行回数
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
            MLE_mu_kent[i] = MLE[0]
            MLE_rho_kent[i] = MLE[1]
            MLE_time_kent[i] = e_time - s_time

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
            method2_mu[i] = est.x[0]
            method2_rho[i] = est.x[1]
            method2_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = est_W2_method3(sample)
            e_time = time.perf_counter()
            method3_mu[i] = est.x[0]
            method3_rho[i] = est.x[1]
            method3_time[i] = e_time - s_time

            # s_time = time.perf_counter()
            # est = est_W1_method3(sample)
            # e_time = time.perf_counter()
            # method4_mu[i] = est.x[0]
            # method4_rho[i] = est.x[1]
            # method4_time[i] = e_time - s_time

        # MSEを計算する
        # MLE_mu_okamura_mse = np.mean((MLE_mu_okamura - true_mu) ** 2)
        # MLE_rho_okamura_mse = np.mean((MLE_rho_okamura - true_rho) ** 2)
        # MLE_time_okamura_mean = np.mean(MLE_time_okamura)
        MLE_mu_kent_mse = np.mean((MLE_mu_kent - true_mu) ** 2)
        MLE_rho_kent_mse = np.mean((MLE_rho_kent - true_rho) ** 2)
        MLE_time_kent_mean = np.mean(MLE_time_kent)
        # MLE_mu_direct_mse = np.mean((MLE_mu_direct - true_mu) ** 2)
        # MLE_rho_direct_mse = np.mean((MLE_rho_direct - true_rho) ** 2)
        # MLE_time_direct_mean = np.mean(MLE_time_direct)
        # method1_mu_mse = np.mean((method1_mu - true_mu) ** 2)
        # method1_rho_mse = np.mean((method1_rho - true_rho) ** 2)
        # method1_time_mean = np.mean(method1_time)
        method2_mu_mse = np.mean((method2_mu - true_mu) ** 2)
        method2_rho_mse = np.mean((method2_rho - true_rho) ** 2)
        method2_time_mean = np.mean(method2_time)
        method3_mu_mse = np.mean((method3_mu - true_mu) ** 2)
        method3_rho_mse = np.mean((method3_rho - true_rho) ** 2)
        method3_time_mean = np.mean(method3_time)
        # method4_mu_mse = np.mean((method4_mu - true_mu) ** 2)
        # method4_rho_mse = np.mean((method4_rho - true_rho) ** 2)
        # method4_time_mean = np.mean(method4_time)

        df.loc[log10_Ns[j]] = [
            np.log10(MLE_mu_kent_mse),
            np.log10(MLE_rho_kent_mse),
            np.log10(method2_mu_mse),
            np.log10(method2_rho_mse),
            np.log10(method3_mu_mse),
            np.log10(method3_rho_mse),
            np.log10(fisher_mat_inv_diag[0]) - log10_Ns[j],
            np.log10(fisher_mat_inv_diag[1]) - log10_Ns[j],
        ]

        # print(
        #     f"MLE by okamura: mu_mse={MLE_mu_okamura_mse}, rho_mse={MLE_rho_okamura_mse}, time={MLE_time_okamura_mean}"
        # )
        print(
            f"MLE by kent: mu_mse={MLE_mu_kent_mse}, rho_mse={MLE_rho_kent_mse}, time={MLE_time_kent_mean}"
        )
        # print(
        #     f"MLE by direct: mu_mse={MLE_mu_direct_mse}, rho_mse={MLE_rho_direct_mse}, time={MLE_time_direct_mean}"
        # )
        # print(
        #     f"W2-est by method1: mu_mse={method1_mu_mse}, rho_mse={method1_rho_mse}, time={method1_time_mean}"
        # )
        print(
            f"W1-est by method2: mu_mse={method2_mu_mse}, rho_mse={method2_rho_mse}, time={method2_time_mean}"
        )
        print(
            f"W2-est by method3: mu_mse={method3_mu_mse}, rho_mse={method3_rho_mse}, time={method3_time_mean}"
        )
        # print(
        #     f"W1-est by method3: mu_mse={method4_mu_mse}, rho_mse={method4_rho_mse}, time={method4_time_mean}"
        # )

    print(df)
    df.to_csv("./data/ex7_wrapcauchy_MSE.csv")


if __name__ == "__main__":
    main()
