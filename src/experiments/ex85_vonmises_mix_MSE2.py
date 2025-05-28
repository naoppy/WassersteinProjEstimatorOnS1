"""
何回かサンプルをとってMSEを計算する
MSE, W2-estimator(method1), W1-estimator(method2)の比較
"""

import time
from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as stats
from parfor import pmap
from scipy import optimize

from ..calc_semidiscrete_W_dist import method1, method2
from ..distributions import vonmises

bounds = ((-np.pi, np.pi), (0.1, 5))


def Wp_cost_func3(x, given_data_normed_sorted, p: int):
    sample = vonmises.fast_quantile_sampling(x[0], x[1], len(given_data_normed_sorted))
    sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
    sample = np.sort(sample)
    return method1.method1(given_data_normed_sorted, sample, p=p, sorted=True)


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
    # return optimize.minimize(
    #     cost_func,
    #     (0, 2.5),
    #     bounds=bounds,
    #     method="powell",
    #     options={"xtol": 1e-6, "ftol": 1e-6},
    # )
    return optimize.differential_evolution(
        cost_func,
        bounds=bounds,
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
    return optimize.differential_evolution(
        cost_func,
        bounds=bounds,
    )


def run_once(i, true_mu, true_kappa, uniform_noise_rate, N: int):
    dists = [
        stats.vonmises(loc=true_mu, kappa=true_kappa),
        stats.uniform(loc=0, scale=2 * np.pi),
    ]
    weights = [1 - uniform_noise_rate, uniform_noise_rate]

    def sample_gen(N):
        draw = np.random.choice([0, 1], N, p=weights)
        sample1 = dists[0].rvs(np.count_nonzero(draw == 0))
        sample2 = dists[1].rvs(np.count_nonzero(draw == 1))
        return np.concatenate([sample1, sample2])

    sample = sample_gen(N)
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
    true_mu = np.pi / 4
    true_kappa = 5
    N = np.power(10, 5).astype(np.int64)
    print(
        f"true mu={true_mu}, true kappa={true_kappa}, N={N}"
    )
    print("(mu, kappa, time)")

    noise_rates = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    try_nums = [1000] * len(noise_rates)

    df = pd.DataFrame(
        index=noise_rates,
        columns=[
            "MLE_mu",
            "MLE_kappa",
            "W1(method2)_mu",
            "W1(method2)_kappa",
            "W2(method3)_mu",
            "W2(method3)_kappa",
        ],
    )

    for j, (noise_rate, try_num) in enumerate(zip(noise_rates, try_nums, strict=True)):
        print(f"noise_rate={noise_rate}")
        MLE_mu = np.zeros(try_num)
        MLE_kappa = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        method1_mu = np.zeros(try_num)
        method1_kappa = np.zeros(try_num)
        method1_time = np.zeros(try_num)
        method2_mu = np.zeros(try_num)
        method2_kappa = np.zeros(try_num)
        method2_time = np.zeros(try_num)

        result = pmap(
            run_once, range(try_num), (true_mu, true_kappa, noise_rate, N)
        )
        for i in range(try_num):
            r = result[i]
            MLE_mu[i] = r[0]
            MLE_kappa[i] = r[1]
            MLE_time[i] = r[2]
            method1_mu[i] = r[3]
            method1_kappa[i] = r[4]
            method1_time[i] = r[5]
            method2_mu[i] = r[6]
            method2_kappa[i] = r[7]
            method2_time[i] = r[8]

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_kappa_mse = np.mean((MLE_kappa - true_kappa) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        W1method2_mu_mse = np.mean((method1_mu - true_mu) ** 2)
        W1method2_kappa_mse = np.mean((method1_kappa - true_kappa) ** 2)
        W1method2_time_mean = np.mean(method1_time)
        W2method3_mu_mse = np.mean((method2_mu - true_mu) ** 2)
        W2method3_kappa_mse = np.mean((method2_kappa - true_kappa) ** 2)
        W2method3_time_mean = np.mean(method2_time)
        df.loc[noise_rate] = [
            np.log10(MLE_mu_mse),
            np.log10(MLE_kappa_mse),
            np.log10(W1method2_mu_mse),
            np.log10(W1method2_kappa_mse),
            np.log10(W2method3_mu_mse),
            np.log10(W2method3_kappa_mse),
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse}, kappa_mse={MLE_kappa_mse}, time={MLE_time_mean}"
        )
        print(
            f"W1method2: mu_mse={W1method2_mu_mse}, kappa_mse={W1method2_kappa_mse}, time={W1method2_time_mean}"
        )
        print(
            f"W2method3: mu_mse={W2method3_mu_mse}, kappa_mse={W2method3_kappa_mse}, time={W2method3_time_mean}"
        )
    print(df)
    df.to_csv("./data/ex85_change_noise_rate.csv")


if __name__ == "__main__":
    main()
