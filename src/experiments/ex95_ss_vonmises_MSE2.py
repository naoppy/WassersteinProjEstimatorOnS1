"""
何回かサンプルをとってMSEを計算する
MLE, W1-estimator(method2)の比較
"""

import time

import numpy as np
import pandas as pd
from numpy import typing as npt
from parfor import pmap
from scipy import optimize

from ..calc_semidiscrete_W_dist import method2
from ..distributions import sine_skewed_vonmises


TOL = 1e-7

bounds = ((-np.pi, np.pi), (0.01, 2), (-1, 1))


def est_method2(given_data):
    """Calc W1-estimator using method1

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = sine_skewed_vonmises.cumsum_hist_data(given_data, bin_num)

    def cost_func(x):
        mu, kappa, lambda_ = x
        dist_cumsum_hist = sine_skewed_vonmises.cumsum_hist(mu, kappa, lambda_, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.differential_evolution(
        cost_func,
        tol=TOL,
        bounds=bounds,
    )


def run_once(i, true_mu, true_kappa, true_lambda, N: int) -> npt.NDArray[np.float64]:
    sample = sine_skewed_vonmises.rejection_sampling(
        N, true_mu, true_kappa, true_lambda
    )

    s_time = time.perf_counter()
    MLE = sine_skewed_vonmises.MLE_direct_opt(sample, bounds=bounds, tol=TOL)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_kappa = MLE[1]
    MLE_lambda = MLE[2]
    MLE_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_method2(sample)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_kappa = est.x[1]
    W1method2_lambda = est.x[2]
    W1method2_time = e_time - s_time

    return np.array(
        [
            MLE_mu,
            MLE_kappa,
            MLE_lambda,
            MLE_time,
            W1method2_mu,
            W1method2_kappa,
            W1method2_lambda,
            W1method2_time,
        ]
    )


def _main():
    true_mu = 0
    true_kappa = 1
    N = np.power(10, 5).astype(np.int64)
    print(f"true mu={true_mu}, true kappa={true_kappa}, N={N}")
    print("(mu, kappa, lambda, time)")

    lambdas = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    try_nums = [10000] * len(lambdas)

    df = pd.DataFrame(
        index=lambdas,
        columns=[
            "MLE_mu",
            "MLE_kappa",
            "MLE_lambda",
            "W1(method2)_mu",
            "W1(method2)_kappa",
            "W1(method2)_lambda",
            "Cramer-Rao Lower Bound of mu",
            "Cramer-Rao Lower Bound of kappa",
            "Cramer-Rao Lower Bound of lambda",
        ],
    )

    for j, (true_lambda, try_num) in enumerate(zip(lambdas, try_nums, strict=True)):
        print(f"true lambda={true_lambda}")
        MLE_mu = np.zeros(try_num)
        MLE_kappa = np.zeros(try_num)
        MLE_lambda = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        W1method2_mu = np.zeros(try_num)
        W1method2_kappa = np.zeros(try_num)
        W1method2_lambda = np.zeros(try_num)
        W1method2_time = np.zeros(try_num)

        fisher_mat_inv_diag = sine_skewed_vonmises.fisher_mat_inv_diag(
            true_kappa, true_lambda
        )

        # MSEをとるための試行回数
        result = pmap(run_once, range(try_num), (true_mu, true_kappa, true_lambda, N))
        for i in range(try_num):
            r = result[i]
            MLE_mu[i] = r[0]
            MLE_kappa[i] = r[1]
            MLE_lambda[i] = r[2]
            MLE_time[i] = r[3]
            W1method2_mu[i] = r[4]
            W1method2_kappa[i] = r[5]
            W1method2_lambda[i] = r[6]
            W1method2_time[i] = r[7]

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_kappa_mse = np.mean((MLE_kappa - true_kappa) ** 2)
        MLE_lambda_mse = np.mean((MLE_lambda - true_lambda) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        W1method2_mu_mse = np.mean((W1method2_mu - true_mu) ** 2)
        W1method2_kappa_mse = np.mean((W1method2_kappa - true_kappa) ** 2)
        W1method2_lambda_mse = np.mean((W1method2_lambda - true_lambda) ** 2)
        W1method2_time_mean = np.mean(W1method2_time)

        df.loc[true_lambda] = [
            np.log10(MLE_mu_mse),
            np.log10(MLE_kappa_mse),
            np.log10(MLE_lambda_mse),
            np.log10(W1method2_mu_mse),
            np.log10(W1method2_kappa_mse),
            np.log10(W1method2_lambda_mse),
            np.log10(fisher_mat_inv_diag[0]) - np.log10(N),
            np.log10(fisher_mat_inv_diag[1]) - np.log10(N),
            np.log10(fisher_mat_inv_diag[2]) - np.log10(N),
        ]

        print("MLE:")
        print(f"{MLE_mu_mse}, {MLE_kappa_mse}, {MLE_lambda_mse}, {MLE_time_mean}")
        print("Method2(W1):")
        print(
            f"{W1method2_mu_mse}, {W1method2_kappa_mse}, {W1method2_lambda_mse}, {W1method2_time_mean}"
        )
        print(df)
        df.to_csv("./data/ex95_change_lambda.csv")
    print(df)
    df.to_csv("./data/ex95_change_lambda.csv")


if __name__ == "__main__":
    _main()
