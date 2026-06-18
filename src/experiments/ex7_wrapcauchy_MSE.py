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

from src.distributions import wrappedcauchy


def run_once(i, true_mu, true_rho, N: int) -> dict:
    # [0, 2*pi] の範囲でサンプリングしたいが、[mu, mu + 2*pi] の範囲になっているので修正
    sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    s_time = time.perf_counter()
    MLE = wrappedcauchy.MLE_Kent(sample, tol=1e-15)
    e_time = time.perf_counter()
    MLE_mu_kent = MLE[0]
    MLE_rho_kent = MLE[1]
    MLE_time_kent = e_time - s_time

    s_time = time.perf_counter()
    est = wrappedcauchy.W1_equal_div(sample, x0=MLE)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_rho = est.x[1]
    W1method2_time = e_time - s_time

    s_time = time.perf_counter()
    est = wrappedcauchy.W2_quantile_sampling(sample, x0=MLE)
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_rho = est.x[1]
    W2method3_time = e_time - s_time

    return {
        "MLE_mu_kent": MLE_mu_kent,
        "MLE_rho_kent": MLE_rho_kent,
        "MLE_time_kent": MLE_time_kent,
        "W1method2_mu": W1method2_mu,
        "W1method2_rho": W1method2_rho,
        "W1method2_time": W1method2_time,
        "W2method3_mu": W2method3_mu,
        "W2method3_rho": W2method3_rho,
        "W2method3_time": W2method3_time,
    }


def main():
    true_mu = np.pi / 8
    # 実験条件1
    true_rho = 0.4
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
            "MLE_rho",
            "W1(method2)_mu",
            "W1(method2)_rho",
            "W2(method3)_mu",
            "W2(method3)_rho",
            "Cramer-Rao Lower Bound of mu",
            "Cramer-Rao Lower Bound of rho",
        ],
    )
    df.index.name = "log10N"
    fisher_mat_inv_diag = wrappedcauchy.fisher_mat_inv_diag(true_rho)

    for j, (N, try_num) in enumerate(
        zip(Ns, try_nums, strict=True)
    ):  # データ数Nを変える
        print(f"N={N}")

        result = Parallel(n_jobs=-1)(
            delayed(run_once)(i, true_mu, true_rho, N)
            for i in tqdm(range(try_num), desc=f"N={N}")
        )
        df_trial = pd.DataFrame(result)

        # MSEを計算する
        MLE_mu_kent_mse = np.mean((df_trial["MLE_mu_kent"] - true_mu) ** 2)
        MLE_rho_kent_mse = np.mean((df_trial["MLE_rho_kent"] - true_rho) ** 2)
        MLE_time_kent_mean = df_trial["MLE_time_kent"].mean()

        method2_mu_mse = np.mean((df_trial["W1method2_mu"] - true_mu) ** 2)
        method2_rho_mse = np.mean((df_trial["W1method2_rho"] - true_rho) ** 2)
        method2_time_mean = df_trial["W1method2_time"].mean()

        method3_mu_mse = np.mean((df_trial["W2method3_mu"] - true_mu) ** 2)
        method3_rho_mse = np.mean((df_trial["W2method3_rho"] - true_rho) ** 2)
        method3_time_mean = df_trial["W2method3_time"].mean()

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

        print(
            f"MLE kent: mu_mse={MLE_mu_kent_mse}, "
            f"rho_mse={MLE_rho_kent_mse}, time={MLE_time_kent_mean}"
        )
        print(
            f"W1 method2: mu_mse={method2_mu_mse}, "
            f"rho_mse={method2_rho_mse}, time={method2_time_mean}"
        )
        print(
            f"W2 method3: mu_mse={method3_mu_mse}, "
            f"rho_mse={method3_rho_mse}, time={method3_time_mean}"
        )

    df.index.name = "log10N"
    print(df)
    df.to_csv("./data/wrapcauchy_MSE/ex7_wrapcauchy_MSE.csv")


if __name__ == "__main__":
    main()
