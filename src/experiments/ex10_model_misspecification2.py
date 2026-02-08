"""
データの生成分布が巻き込みコーシー分布のときに、フォンミーゼス分布でフィッティングした際に平均と円周分散がどうなるかを調べる。
MLE, W1, W2で比較。
つまり、model-misspecificationの意味でのロバスト性を調べる。
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
from ..distributions import vonmises, wrapedcauchy

bounds = ((-np.pi, np.pi), (0.1, 5))


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
        (0, 2.5),
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
        (0, 2.5),
        bounds=bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def run_once(i, true_mu, true_rho, N: int) -> npt.NDArray[np.float64]:
    # データは巻き込みコーシー分布、モデルはフォンミーゼス分布
    sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    s_time = time.perf_counter()
    MLE = vonmises.MLE(vonmises.T(sample), N)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_CV = vonmises.circular_variance(MLE[1])
    MLE_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_W1_method2(sample)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_CV = vonmises.circular_variance(est.x[1])
    W1method2_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_W1_method3(sample)
    e_time = time.perf_counter()
    W1method3_mu = est.x[0]
    W1method3_CV = vonmises.circular_variance(est.x[1])
    W1method3_time = e_time - s_time

    s_time = time.perf_counter()
    est = est_W2_method3(sample)
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_CV = vonmises.circular_variance(est.x[1])
    W2method3_time = e_time - s_time

    return np.array(
        [
            MLE_mu,
            MLE_CV,
            MLE_time,
            W1method2_mu,
            W1method2_CV,
            W1method2_time,
            W1method3_mu,
            W1method3_CV,
            W1method3_time,
            W2method3_mu,
            W2method3_CV,
            W2method3_time,
        ]
    )


def main():
    # true distribution is wraped cauchy
    true_mu = np.pi / 8
    true_rho = 0.3
    true_CV = wrapedcauchy.circular_variance(true_rho)
    print(f"true mu={true_mu}, true rho={true_rho}, true_CV={true_CV}")
    print("(mu, CV, time)")

    log10_Ns = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    Ns = np.power(10, log10_Ns).astype(np.int64)
    try_nums = [1000] * len(Ns)

    df = pd.DataFrame(
        index=log10_Ns,
        columns=[
            "MLE_mu",
            "MLE_CV",
            "W1(method2)_mu",
            "W1(method2)_CV",
            "W1(method3)_mu",
            "W1(method3)_CV",
            "W2(method3)_mu",
            "W2(method3)_CV",
        ],
    )

    for j, (N, try_num) in enumerate(
        zip(Ns, try_nums, strict=True)
    ):  # データ数Nを変える
        print(f"N={N}")
        MLE_mu = np.zeros(try_num)
        MLE_CV = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        W1method2_mu = np.zeros(try_num)
        W1method2_CV = np.zeros(try_num)
        W1method2_time = np.zeros(try_num)
        W1method3_mu = np.zeros(try_num)
        W1method3_CV = np.zeros(try_num)
        W1method3_time = np.zeros(try_num)
        W2method3_mu = np.zeros(try_num)
        W2method3_CV = np.zeros(try_num)
        W2method3_time = np.zeros(try_num)

        # MSEをとるための試行回数
        result = pmap(run_once, range(try_num), (true_mu, true_rho, N))
        for i in range(try_num):
            r = result[i]
            MLE_mu[i] = r[0]
            MLE_CV[i] = r[1]
            MLE_time[i] = r[2]
            W1method2_mu[i] = r[3]
            W1method2_CV[i] = r[4]
            W1method2_time[i] = r[5]
            W1method3_mu[i] = r[6]
            W1method3_CV[i] = r[7]
            W1method3_time[i] = r[8]
            W2method3_mu[i] = r[9]
            W2method3_CV[i] = r[10]
            W2method3_time[i] = r[11]

        # MSEを計算する
        MLE_mu_mse = np.mean((MLE_mu - true_mu) ** 2)
        MLE_CV_mse = np.mean((MLE_CV - true_CV) ** 2)
        MLE_time_mean = np.mean(MLE_time)
        W1method2_mu_mse = np.mean((W1method2_mu - true_mu) ** 2)
        W1method2_CV_mse = np.mean((W1method2_CV - true_CV) ** 2)
        W1method2_time_mean = np.mean(W1method2_time)
        W1method3_mu_mse = np.mean((W1method3_mu - true_mu) ** 2)
        W1method3_CV_mse = np.mean((W1method3_CV - true_CV) ** 2)
        W1method3_time_mean = np.mean(W1method3_time)
        W2method3_mu_mse = np.mean((W2method3_mu - true_mu) ** 2)
        W2method3_CV_mse = np.mean((W2method3_CV - true_CV) ** 2)
        W2method3_time_mean = np.mean(W2method3_time)
        df.loc[log10_Ns[j]] = [
            np.log10(MLE_mu_mse),
            np.log10(MLE_CV_mse),
            np.log10(W1method2_mu_mse),
            np.log10(W1method2_CV_mse),
            np.log10(W1method3_mu_mse),
            np.log10(W1method3_CV_mse),
            np.log10(W2method3_mu_mse),
            np.log10(W2method3_CV_mse),
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse}, CV_mse={MLE_CV_mse}, time={MLE_time_mean}"
        )
        print(
            f"W1 method2: mu_mse={W1method2_mu_mse}, CV_mse={W1method2_CV_mse}, time={W1method2_time_mean}"
        )
        print(
            f"W1 method3: mu_mse={W1method3_mu_mse}, CV_mse={W1method3_CV_mse}, time={W1method3_time_mean}"
        )
        print(
            f"W2 method3: mu_mse={W2method3_mu_mse}, CV_mse={W2method3_CV_mse}, time={W2method3_time_mean}"
        )

    print(df)
    df.to_csv("./data/ex10_model_misspecification2_MSE.csv")


if __name__ == "__main__":
    main()
