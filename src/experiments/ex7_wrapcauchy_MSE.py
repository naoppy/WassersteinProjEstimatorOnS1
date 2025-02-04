"""
何回かサンプルをとってMSEを計算する
MSE, W2-estimator(method1), W1-estimator(method2)の比較
"""

import time
from functools import partial

import numpy as np
import scipy.stats as stats
from scipy import optimize
from tqdm import tqdm

from ..calc_semidiscrete_W_dist import method1, method2
from ..cauchy import (
    MLE_wrapcauchy_Kent_method,
    MLE_wrapped_cauchy_OKAMURA_method,
    wrapped_cauchy_cumsum_hist,
)


def W2_cost_func(x, given_data_normed_sorted):
    sample = stats.wrapcauchy(loc=x[0], c=x[1]).rvs(len(given_data_normed_sorted))
    sample = np.remainder(sample, 2 * np.pi)
    sample = np.sort(sample)
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def est_method1(given_data):
    """Calc W2-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(W2_cost_func, given_data_normed_sorted=given_data_norm_sorted)
    bounds = ((0, 2 * np.pi), (0.05, 0.95))
    finish_func = partial(optimize.minimize, method="powell", bounds=bounds)

    return optimize.brute(
        cost_func,
        bounds,
        full_output=True,
        finish=finish_func,
        Ns=100,
        workers=-1,
    )


def est_method2(given_data):
    """Calc W1-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist_data(given_data, bin_num)

    def cost_func(x):
        mu, rho = x
        dist_cumsum_hist = wrapped_cauchy_cumsum_hist.cumsum_hist(mu, rho, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.minimize(
        cost_func,
        (np.pi, 0.5),
        bounds=((0, 2 * np.pi), (0.05, 0.95)),
        # for powell method
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

    Ns = [100, 500, 1000, 5000, 10000]
    # Ns = [1000]
    try_nums = [100, 100, 100, 100, 100]
    # try_nums = [10]
    for N, try_num in zip(Ns, try_nums, strict=True):  # データ数Nを変える
        print(f"N={N}")
        MLE_mu_okamura = np.zeros(try_num)
        MLE_rho_okamura = np.zeros(try_num)
        MLE_time_okamura = np.zeros(try_num)
        MLE_mu_kent = np.zeros(try_num)
        MLE_rho_kent = np.zeros(try_num)
        MLE_time_kent = np.zeros(try_num)
        method1_mu = np.zeros(try_num)
        method1_rho = np.zeros(try_num)
        method1_time = np.zeros(try_num)
        method2_mu = np.zeros(try_num)
        method2_rho = np.zeros(try_num)
        method2_time = np.zeros(try_num)

        for i in tqdm(range(try_num)):  # MSEをとるための試行回数
            # [0, 2*pi] の範囲でサンプリングしたいが、[mu, mu + 2*pi] の範囲になっているので修正
            sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
            sample = np.remainder(sample, 2 * np.pi)

            s_time = time.perf_counter()
            MLE = MLE_wrapped_cauchy_OKAMURA_method.calc_MLE(sample, N, iter_num=10000)
            e_time = time.perf_counter()
            MLE_mu_okamura[i] = np.angle(MLE)
            MLE_rho_okamura[i] = np.abs(MLE)
            MLE_time_okamura[i] = e_time - s_time

            s_time = time.perf_counter()
            MLE = MLE_wrapcauchy_Kent_method.calc_MLE(sample, tol=1e-15)
            e_time = time.perf_counter()
            MLE_mu_kent[i] = MLE[0]
            MLE_rho_kent[i] = MLE[1]
            MLE_time_kent[i] = e_time - s_time

            s_time = time.perf_counter()
            # est = est_method1(sample)
            e_time = time.perf_counter()
            # method1_mu[i] = est[0][0]
            # method1_rho[i] = est[0][1]
            method1_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = est_method2(sample)
            e_time = time.perf_counter()
            method2_mu[i] = est.x[0]
            method2_rho[i] = est.x[1]
            method2_time[i] = e_time - s_time

        # MSEを計算する
        MLE_mu_okamura_mse = np.mean((MLE_mu_okamura - true_mu) ** 2)
        MLE_kappa_okamura_mse = np.mean((MLE_rho_okamura - true_rho) ** 2)
        MLE_time_okamura_mean = np.mean(MLE_time_okamura)
        MLE_mu_kent_mse = np.mean((MLE_mu_kent - true_mu) ** 2)
        MLE_kappa_kent_mse = np.mean((MLE_rho_kent - true_rho) ** 2)
        MLE_time_kent_mean = np.mean(MLE_time_kent)
        method1_mu_mse = np.mean((method1_mu - true_mu) ** 2)
        method1_kappa_mse = np.mean((method1_rho - true_rho) ** 2)
        method1_time_mean = np.mean(method1_time)
        method2_mu_mse = np.mean((method2_mu - true_mu) ** 2)
        method2_kappa_mse = np.mean((method2_rho - true_rho) ** 2)
        method2_time_mean = np.mean(method2_time)

        print(
            f"MLE by okamura: mu_mse={MLE_mu_okamura_mse}, rho_mse={MLE_kappa_okamura_mse}, time={MLE_time_okamura_mean}"
        )
        print(
            f"MLE by kent: mu_mse={MLE_mu_kent_mse}, rho_mse={MLE_kappa_kent_mse}, time={MLE_time_kent_mean}"
        )
        print(
            f"W2-est by method1: mu_mse={method1_mu_mse}, rho_mse={method1_kappa_mse}, time={method1_time_mean}"
        )
        print(
            f"W1-est by method2: mu_mse={method2_mu_mse}, rho_mse={method2_kappa_mse}, time={method2_time_mean}"
        )


if __name__ == "__main__":
    main()
