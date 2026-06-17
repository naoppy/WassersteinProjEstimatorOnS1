"""
何回かサンプルをとってMSEを計算する
MSE, W2-estimator(method1), W1-estimator(method2)の比較
"""

import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from tqdm import tqdm

from src.distributions import vonmises


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
    MLE = vonmises.MLE_direct(sample)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_kappa = MLE[1]
    MLE_time = e_time - s_time

    s_time = time.perf_counter()
    est = vonmises.W1_equal_div(sample, method="differential_evolution")
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_kappa = est.x[1]
    W1method2_time = e_time - s_time

    s_time = time.perf_counter()
    est = vonmises.W2_quantile_sampling(sample, method="differential_evolution")
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_kappa = est.x[1]
    W2method3_time = e_time - s_time

    s_time = time.perf_counter()
    est = vonmises.type0_estimate(sample, gamma=0.5)
    e_time = time.perf_counter()
    type0_gamma05_mu = est[0]
    type0_gamma05_kappa = est[1]
    type0_gamma05_time = e_time - s_time

    s_time = time.perf_counter()
    est = vonmises.type1_estimate(sample, beta=0.5)
    e_time = time.perf_counter()
    type1_beta05_mu = est[0]
    type1_beta05_kappa = est[1]
    type1_beta05_time = e_time - s_time

    return {
        "MLE_mu": MLE_mu,
        "MLE_kappa": MLE_kappa,
        "MLE_time": MLE_time,
        "W1method2_mu": W1method2_mu,
        "W1method2_kappa": W1method2_kappa,
        "W1method2_time": W1method2_time,
        "W2method3_mu": W2method3_mu,
        "W2method3_kappa": W2method3_kappa,
        "W2method3_time": W2method3_time,
        "type0_gamma05_mu": type0_gamma05_mu,
        "type0_gamma05_kappa": type0_gamma05_kappa,
        "type0_gamma05_time": type0_gamma05_time,
        "type1_beta05_mu": type1_beta05_mu,
        "type1_beta05_kappa": type1_beta05_kappa,
        "type1_beta05_time": type1_beta05_time,
    }


def main():
    true_mu = np.pi / 4
    true_kappa = 5
    uniform_noise_rate = 0.1
    print(
        f"true mu={true_mu}, true kappa={true_kappa}, "
        f"uniform noise rate={uniform_noise_rate}"
    )
    print("(mu, kappa, time)")

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
            "type0_gamma05_mu",
            "type0_gamma05_kappa",
            "type1_beta05_mu",
            "type1_beta05_kappa",
        ],
    )
    df.index.name = "log10N"

    for _j, (N, try_num) in enumerate(zip(Ns, try_nums, strict=True)):
        print(f"N={N}")

        result = Parallel(n_jobs=-1)(
            delayed(run_once)(i, true_mu, true_kappa, uniform_noise_rate, N)
            for i in tqdm(range(try_num), desc=f"N={N}")
        )
        df_trial = pd.DataFrame(result)

        # MSEを計算する
        MLE_mu_mse = np.mean((df_trial["MLE_mu"] - true_mu) ** 2)
        MLE_kappa_mse = np.mean((df_trial["MLE_kappa"] - true_kappa) ** 2)
        MLE_time_mean = df_trial["MLE_time"].mean()

        W1method2_mu_mse = np.mean((df_trial["W1method2_mu"] - true_mu) ** 2)
        W1method2_kappa_mse = np.mean((df_trial["W1method2_kappa"] - true_kappa) ** 2)
        W1method2_time_mean = df_trial["W1method2_time"].mean()

        W2method3_mu_mse = np.mean((df_trial["W2method3_mu"] - true_mu) ** 2)
        W2method3_kappa_mse = np.mean((df_trial["W2method3_kappa"] - true_kappa) ** 2)
        W2method3_time_mean = df_trial["W2method3_time"].mean()

        type0_gamma05_mu_mse = np.mean((df_trial["type0_gamma05_mu"] - true_mu) ** 2)
        type0_gamma05_kappa_mse = np.mean(
            (df_trial["type0_gamma05_kappa"] - true_kappa) ** 2
        )
        type0_gamma05_time_mean = df_trial["type0_gamma05_time"].mean()

        type1_beta05_mu_mse = np.mean((df_trial["type1_beta05_mu"] - true_mu) ** 2)
        type1_beta05_kappa_mse = np.mean(
            (df_trial["type1_beta05_kappa"] - true_kappa) ** 2
        )
        type1_beta05_time_mean = df_trial["type1_beta05_time"].mean()

        df.loc[log10_Ns[_j]] = [
            np.log10(MLE_mu_mse),
            np.log10(MLE_kappa_mse),
            np.log10(W1method2_mu_mse),
            np.log10(W1method2_kappa_mse),
            np.log10(W2method3_mu_mse),
            np.log10(W2method3_kappa_mse),
            np.log10(type0_gamma05_mu_mse),
            np.log10(type0_gamma05_kappa_mse),
            np.log10(type1_beta05_mu_mse),
            np.log10(type1_beta05_kappa_mse),
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse}, kappa_mse={MLE_kappa_mse}, time={MLE_time_mean}"
        )
        print(
            f"W1method2: mu_mse={W1method2_mu_mse}, "
            f"kappa_mse={W1method2_kappa_mse}, "
            f"time={W1method2_time_mean}"
        )
        print(
            f"W2method3: mu_mse={W2method3_mu_mse}, "
            f"kappa_mse={W2method3_kappa_mse}, "
            f"time={W2method3_time_mean}"
        )
        print(
            f"type0_gamma05: mu_mse={type0_gamma05_mu_mse}, "
            f"kappa_mse={type0_gamma05_kappa_mse}, "
            f"time={type0_gamma05_time_mean}"
        )
        print(
            f"type1_beta05: mu_mse={type1_beta05_mu_mse}, "
            f"kappa_mse={type1_beta05_kappa_mse}, "
            f"time={type1_beta05_time_mean}"
        )
    print(df)
    df.to_csv("./data/ex8_vonmises_mix_MSE.csv")


if __name__ == "__main__":
    main()
