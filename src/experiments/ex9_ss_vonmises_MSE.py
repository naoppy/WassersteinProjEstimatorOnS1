"""
何回かサンプルをとってMSEを計算する
MLE, W1-estimator(method2)の比較
"""

import time

import numpy as np
from scipy import optimize
from tqdm import tqdm

from ..calc_semidiscrete_W_dist import method2
from ..distributions import sine_skewed_vonmises


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
        tol=0.001,
        bounds=((-np.pi, np.pi), (0.01, 4), (-1, 1)),
    )


def main():
    true_mu = 0
    true_kappa = 1
    true_lambda = 0.7
    print(
        f"true mu={true_mu}, true kappa={true_kappa}, true lambda={true_lambda}"
    )

    Ns = np.power(10, [2, 2.5, 3, 3.5, 4, 4.5, 5]).astype(np.int64)
    try_nums = [100, 100, 100, 100, 100, 50, 25]

    for N, try_num in zip(Ns, try_nums, strict=True):  # データ数Nを変える
        print(f"N={N}")
        MLE_mu = np.zeros(try_num)
        MLE_kappa = np.zeros(try_num)
        MLE_lambda = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        method1_mu = np.zeros(try_num)
        method1_kappa = np.zeros(try_num)
        method1_lambda = np.zeros(try_num)
        method1_time = np.zeros(try_num)
        method2_mu = np.zeros(try_num)
        method2_kappa = np.zeros(try_num)
        method2_lambda = np.zeros(try_num)
        method2_time = np.zeros(try_num)

        for i in tqdm(range(try_num)):  # MSEをとるための試行回数
            sample = sine_skewed_vonmises.rejection_sampling(N, true_mu, true_kappa, true_lambda)

            s_time = time.perf_counter()
            MLE = sine_skewed_vonmises.MLE_direct_opt(sample)
            e_time = time.perf_counter()
            MLE_mu[i] = MLE[0]
            MLE_kappa[i] = MLE[1]
            MLE_lambda[i] = MLE[2]
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
            method2_lambda[i] = est.x[2]
            method2_time[i] = e_time - s_time

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_kappa_mse = np.mean((MLE_kappa - true_kappa) ** 2)
        MLE_lambda_mse = np.mean((MLE_lambda - true_lambda) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        method1_mu_mse = np.mean((method1_mu - true_mu) ** 2)
        method1_kappa_mse = np.mean((method1_kappa - true_kappa) ** 2)
        method1_lambda_mse = np.mean((method1_lambda - true_lambda) ** 2)
        method1_time_mean = np.mean(method1_time)
        method2_mu_mse = np.mean((method2_mu - true_mu) ** 2)
        method2_kappa_mse = np.mean((method2_kappa - true_kappa) ** 2)
        method2_lambda_mse = np.mean((method2_lambda - true_lambda) ** 2)
        method2_time_mean = np.mean(method2_time)

        print("MLE: (mu, kappa, lambda, time)")
        print(
            f"{MLE_mu_mse}, {MLE_kappa_mse}, {MLE_lambda_mse}, {MLE_time_mean}"
        )
        print("Method1(W2):")
        print(
            f"{method1_mu_mse}, {method1_kappa_mse}, {method1_lambda_mse}, {method1_time_mean}"
        )
        print("Method2(W1):")
        print(
            f"{method2_mu_mse}, {method2_kappa_mse}, {method2_lambda_mse}, {method2_time_mean}"
        )


if __name__ == "__main__":
    main()
