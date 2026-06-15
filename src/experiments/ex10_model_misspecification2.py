"""
データの生成分布が巻き込みコーシー分布のときに、フォンミーゼス分布でフィッティングした際に平均と円周分散がどうなるかを調べる。
MLE, W1, W2で比較。
つまり、model-misspecificationの意味でのロバスト性を調べる。
"""

import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy import typing as npt
from parfor import pmap

from src.distributions import vonmises, wrappedcauchy
from src.utils import dist_utils


def run_once(i, true_mu, true_rho, N: int) -> npt.NDArray[np.float64]:
    # データは巻き込みコーシー分布、モデルはフォンミーゼス分布
    sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    def p_pdf(theta):
        return dist_utils.wrapcauchy_pdf(theta, true_mu, true_rho)

    def p_cdf(theta):
        return wrappedcauchy.wrapcauchy_periodic_cdf_analytical(
            theta, true_rho, true_mu
        ) - wrappedcauchy.wrapcauchy_periodic_cdf_analytical(0, true_rho, true_mu)

    # MLE
    s_time = time.perf_counter()
    MLE = vonmises.MLE_direct(sample)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_kappa = MLE[1]
    MLE_time = e_time - s_time

    def q_pdf_mle(theta):
        return dist_utils.vonmises_pdf(theta, MLE_mu, MLE_kappa)

    dist_q_mle = stats.vonmises(loc=MLE_mu, kappa=MLE_kappa)

    def q_cdf_mle(theta):
        return dist_q_mle.cdf(theta) - dist_q_mle.cdf(0)

    mle_kl, mle_w1, mle_w2 = dist_utils.calculate_distances(
        p_pdf, q_pdf_mle, p_cdf=p_cdf, q_cdf=q_cdf_mle
    )

    # W1 method2 (equal division)
    s_time = time.perf_counter()
    est = vonmises.W1_equal_div(sample, x0=MLE)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_kappa = est.x[1]
    W1method2_time = e_time - s_time

    def q_pdf_w1m2(theta):
        return dist_utils.vonmises_pdf(theta, W1method2_mu, W1method2_kappa)

    dist_q_w1m2 = stats.vonmises(loc=W1method2_mu, kappa=W1method2_kappa)

    def q_cdf_w1m2(theta):
        return dist_q_w1m2.cdf(theta) - dist_q_w1m2.cdf(0)

    w1m2_kl, w1m2_w1, w1m2_w2 = dist_utils.calculate_distances(
        p_pdf, q_pdf_w1m2, p_cdf=p_cdf, q_cdf=q_cdf_w1m2
    )

    # W1 method3 (quantile sampling)
    s_time = time.perf_counter()
    est = vonmises.W1_quantile_sampling(sample, x0=MLE)
    e_time = time.perf_counter()
    W1method3_mu = est.x[0]
    W1method3_kappa = est.x[1]
    W1method3_time = e_time - s_time

    def q_pdf_w1m3(theta):
        return dist_utils.vonmises_pdf(theta, W1method3_mu, W1method3_kappa)

    dist_q_w1m3 = stats.vonmises(loc=W1method3_mu, kappa=W1method3_kappa)

    def q_cdf_w1m3(theta):
        return dist_q_w1m3.cdf(theta) - dist_q_w1m3.cdf(0)

    w1m3_kl, w1m3_w1, w1m3_w2 = dist_utils.calculate_distances(
        p_pdf, q_pdf_w1m3, p_cdf=p_cdf, q_cdf=q_cdf_w1m3
    )

    # W2 method3 (quantile sampling)
    s_time = time.perf_counter()
    est = vonmises.W2_quantile_sampling(sample, x0=MLE)
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_kappa = est.x[1]
    W2method3_time = e_time - s_time

    def q_pdf_w2m3(theta):
        return dist_utils.vonmises_pdf(theta, W2method3_mu, W2method3_kappa)

    dist_q_w2m3 = stats.vonmises(loc=W2method3_mu, kappa=W2method3_kappa)

    def q_cdf_w2m3(theta):
        return dist_q_w2m3.cdf(theta) - dist_q_w2m3.cdf(0)

    w2m3_kl, w2m3_w1, w2m3_w2 = dist_utils.calculate_distances(
        p_pdf, q_pdf_w2m3, p_cdf=p_cdf, q_cdf=q_cdf_w2m3
    )

    return np.array(
        [
            MLE_mu,
            mle_kl,
            mle_w1,
            mle_w2,
            MLE_time,
            W1method2_mu,
            w1m2_kl,
            w1m2_w1,
            w1m2_w2,
            W1method2_time,
            W1method3_mu,
            w1m3_kl,
            w1m3_w1,
            w1m3_w2,
            W1method3_time,
            W2method3_mu,
            w2m3_kl,
            w2m3_w1,
            w2m3_w2,
            W2method3_time,
        ]
    )


def main():
    # true distribution is wraped cauchy
    true_mu = np.pi / 8
    true_rho = 0.75
    print(f"true mu={true_mu}, true rho={true_rho}")
    print("(mu_mse, KL, W1, W2, time)")

    log10_Ns = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    Ns = np.power(10, log10_Ns).astype(np.int64)
    try_nums = [1000] * len(Ns)

    df = pd.DataFrame(
        index=log10_Ns,
        columns=[
            "MLE_mu_mse",
            "MLE_KL",
            "MLE_W1",
            "MLE_W2",
            "W1(method2)_mu_mse",
            "W1(method2)_KL",
            "W1(method2)_W1",
            "W1(method2)_W2",
            "W1(method3)_mu_mse",
            "W1(method3)_KL",
            "W1(method3)_W1",
            "W1(method3)_W2",
            "W2(method3)_mu_mse",
            "W2(method3)_KL",
            "W2(method3)_W1",
            "W2(method3)_W2",
        ],
    )

    for j, (N, try_num) in enumerate(
        zip(Ns, try_nums, strict=True)
    ):  # データ数Nを変える
        print(f"N={N}")
        MLE_mu = np.zeros(try_num)
        MLE_kl = np.zeros(try_num)
        MLE_w1 = np.zeros(try_num)
        MLE_w2 = np.zeros(try_num)
        MLE_time = np.zeros(try_num)
        W1method2_mu = np.zeros(try_num)
        W1method2_kl = np.zeros(try_num)
        W1method2_w1 = np.zeros(try_num)
        W1method2_w2 = np.zeros(try_num)
        W1method2_time = np.zeros(try_num)
        W1method3_mu = np.zeros(try_num)
        W1method3_kl = np.zeros(try_num)
        W1method3_w1 = np.zeros(try_num)
        W1method3_w2 = np.zeros(try_num)
        W1method3_time = np.zeros(try_num)
        W2method3_mu = np.zeros(try_num)
        W2method3_kl = np.zeros(try_num)
        W2method3_w1 = np.zeros(try_num)
        W2method3_w2 = np.zeros(try_num)
        W2method3_time = np.zeros(try_num)

        # MSEをとるための試行回数
        result = pmap(run_once, range(try_num), (true_mu, true_rho, N))
        for i in range(try_num):
            r = result[i]
            MLE_mu[i] = r[0]
            MLE_kl[i] = r[1]
            MLE_w1[i] = r[2]
            MLE_w2[i] = r[3]
            MLE_time[i] = r[4]
            W1method2_mu[i] = r[5]
            W1method2_kl[i] = r[6]
            W1method2_w1[i] = r[7]
            W1method2_w2[i] = r[8]
            W1method2_time[i] = r[9]
            W1method3_mu[i] = r[10]
            W1method3_kl[i] = r[11]
            W1method3_w1[i] = r[12]
            W1method3_w2[i] = r[13]
            W1method3_time[i] = r[14]
            W2method3_mu[i] = r[15]
            W2method3_kl[i] = r[16]
            W2method3_w1[i] = r[17]
            W2method3_w2[i] = r[18]
            W2method3_time[i] = r[19]

        # MSEを計算する
        MLE_mu_mse = np.mean(
            (np.remainder(MLE_mu - true_mu + np.pi, 2 * np.pi) - np.pi) ** 2
        )
        MLE_kl_mean = np.mean(MLE_kl)
        MLE_w1_mean = np.mean(MLE_w1)
        MLE_w2_mean = np.mean(MLE_w2)
        MLE_time_mean = np.mean(MLE_time)

        W1method2_mu_mse = np.mean(
            (np.remainder(W1method2_mu - true_mu + np.pi, 2 * np.pi) - np.pi) ** 2
        )
        W1method2_kl_mean = np.mean(W1method2_kl)
        W1method2_w1_mean = np.mean(W1method2_w1)
        W1method2_w2_mean = np.mean(W1method2_w2)
        W1method2_time_mean = np.mean(W1method2_time)

        W1method3_mu_mse = np.mean(
            (np.remainder(W1method3_mu - true_mu + np.pi, 2 * np.pi) - np.pi) ** 2
        )
        W1method3_kl_mean = np.mean(W1method3_kl)
        W1method3_w1_mean = np.mean(W1method3_w1)
        W1method3_w2_mean = np.mean(W1method3_w2)
        W1method3_time_mean = np.mean(W1method3_time)

        W2method3_mu_mse = np.mean(
            (np.remainder(W2method3_mu - true_mu + np.pi, 2 * np.pi) - np.pi) ** 2
        )
        W2method3_kl_mean = np.mean(W2method3_kl)
        W2method3_w1_mean = np.mean(W2method3_w1)
        W2method3_w2_mean = np.mean(W2method3_w2)
        W2method3_time_mean = np.mean(W2method3_time)

        df.loc[log10_Ns[j]] = [
            MLE_mu_mse,
            MLE_kl_mean,
            MLE_w1_mean,
            MLE_w2_mean,
            W1method2_mu_mse,
            W1method2_kl_mean,
            W1method2_w1_mean,
            W1method2_w2_mean,
            W1method3_mu_mse,
            W1method3_kl_mean,
            W1method3_w1_mean,
            W1method3_w2_mean,
            W2method3_mu_mse,
            W2method3_kl_mean,
            W2method3_w1_mean,
            W2method3_w2_mean,
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse}, KL={MLE_kl_mean}, "
            f"W1={MLE_w1_mean}, W2={MLE_w2_mean}, time={MLE_time_mean}"
        )
        print(
            f"W1 method2: mu_mse={W1method2_mu_mse}, KL={W1method2_kl_mean}, "
            f"W1={W1method2_w1_mean}, W2={W1method2_w2_mean}, "
            f"time={W1method2_time_mean}"
        )
        print(
            f"W1 method3: mu_mse={W1method3_mu_mse}, KL={W1method3_kl_mean}, "
            f"W1={W1method3_w1_mean}, W2={W1method3_w2_mean}, "
            f"time={W1method3_time_mean}"
        )
        print(
            f"W2 method3: mu_mse={W2method3_mu_mse}, KL={W2method3_kl_mean}, "
            f"W1={W2method3_w1_mean}, W2={W2method3_w2_mean}, "
            f"time={W2method3_time_mean}"
        )

    print(df)
    df.to_csv("./data/ex10_model_misspecification2_MSE.csv")


if __name__ == "__main__":
    main()
