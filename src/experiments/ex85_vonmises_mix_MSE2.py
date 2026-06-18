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


def run_once(i, true_mu, true_kappa, uniform_noise_rate, N: int) -> dict:
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
    }


def main():
    true_mu = np.pi / 4
    true_kappa = 5
    N = np.power(10, 5).astype(np.int64)
    print(f"true mu={true_mu}, true kappa={true_kappa}, N={N}")
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
    df.index.name = "noise_rate"

    # noise_rateを変える
    for _j, (noise_rate, try_num) in enumerate(zip(noise_rates, try_nums, strict=True)):
        print(f"noise_rate={noise_rate}")

        result = Parallel(n_jobs=-1)(
            delayed(run_once)(i, true_mu, true_kappa, noise_rate, N)
            for i in tqdm(range(try_num), desc=f"noise_rate={noise_rate:.2f}")
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
            f"W1method2: mu_mse={W1method2_mu_mse}, "
            f"kappa_mse={W1method2_kappa_mse}, "
            f"time={W1method2_time_mean}"
        )
        print(
            f"W2method3: mu_mse={W2method3_mu_mse}, "
            f"kappa_mse={W2method3_kappa_mse}, "
            f"time={W2method3_time_mean}"
        )
        df.index.name = "noise_rate"
        df.to_csv("./data/vonmises_mix/ex85_change_noise_rate.csv")
    df.index.name = "noise_rate"
    print(df)
    df.to_csv("./data/vonmises_mix/ex85_change_noise_rate.csv")


if __name__ == "__main__":
    main()
