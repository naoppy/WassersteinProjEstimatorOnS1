"""
何回かサンプルをとってMSEを計算する
MLE, W1-estimator(method2)の比較
"""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.distributions import sine_skewed_vonmises

TOL = 1e-7
bounds = ((0, 2 * np.pi), (0.01, 2), (-1, 1))


def run_once(i, true_mu, true_kappa, true_lambda, N: int) -> dict:
    sample = sine_skewed_vonmises.rejection_sampling(
        N, true_mu, true_kappa, true_lambda
    )

    s_time = time.perf_counter()
    MLE = sine_skewed_vonmises.MLE_direct(sample, bounds=bounds, tol=TOL)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_kappa = MLE[1]
    MLE_lambda = MLE[2]
    MLE_time = e_time - s_time

    s_time = time.perf_counter()
    est = sine_skewed_vonmises.W1_equal_div(sample, bounds=bounds, tol=TOL)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_kappa = est.x[1]
    W1method2_lambda = est.x[2]
    W1method2_time = e_time - s_time

    return {
        "MLE_mu": MLE_mu,
        "MLE_kappa": MLE_kappa,
        "MLE_lambda": MLE_lambda,
        "MLE_time": MLE_time,
        "W1method2_mu": W1method2_mu,
        "W1method2_kappa": W1method2_kappa,
        "W1method2_lambda": W1method2_lambda,
        "W1method2_time": W1method2_time,
    }


def _main():
    true_mu = 0
    true_kappa = 1
    true_lambda = 0.7
    print(f"true mu={true_mu}, true kappa={true_kappa}, true lambda={true_lambda}")
    print("(mu, kappa, lambda, time)")

    log10_Ns = np.array([2, 2.5, 3, 3.5, 4, 4.5, 5])
    Ns = np.power(10, log10_Ns).astype(np.int64)
    try_nums = [10000] * len(Ns)

    df = pd.DataFrame(
        index=log10_Ns,
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
    df.index.name = "log10N"

    fisher_mat_inv_diag = sine_skewed_vonmises.fisher_mat_inv_diag(
        true_kappa, true_lambda
    )

    for _j, (N, try_num) in enumerate(zip(Ns, try_nums, strict=True)):
        print(f"N={N}")

        # MSEをとるための試行回数
        result = Parallel(n_jobs=-1)(
            delayed(run_once)(i, true_mu, true_kappa, true_lambda, N)
            for i in tqdm(range(try_num), desc=f"N={N}")
        )
        df_trial = pd.DataFrame(result)

        # MSEを計算する
        MLE_mu_mse = np.mean((df_trial["MLE_mu"] - true_mu) ** 2)
        MLE_kappa_mse = np.mean((df_trial["MLE_kappa"] - true_kappa) ** 2)
        MLE_lambda_mse = np.mean((df_trial["MLE_lambda"] - true_lambda) ** 2)
        MLE_time_mean = df_trial["MLE_time"].mean()

        W1method2_mu_mse = np.mean((df_trial["W1method2_mu"] - true_mu) ** 2)
        W1method2_kappa_mse = np.mean((df_trial["W1method2_kappa"] - true_kappa) ** 2)
        W1method2_lambda_mse = np.mean(
            (df_trial["W1method2_lambda"] - true_lambda) ** 2
        )
        W1method2_time_mean = df_trial["W1method2_time"].mean()

        df.loc[log10_Ns[_j]] = [
            np.log10(MLE_mu_mse),
            np.log10(MLE_kappa_mse),
            np.log10(MLE_lambda_mse),
            np.log10(W1method2_mu_mse),
            np.log10(W1method2_kappa_mse),
            np.log10(W1method2_lambda_mse),
            np.log10(fisher_mat_inv_diag[0]) - log10_Ns[_j],
            np.log10(fisher_mat_inv_diag[1]) - log10_Ns[_j],
            np.log10(fisher_mat_inv_diag[2]) - log10_Ns[_j],
        ]

        print("MLE:")
        print(f"{MLE_mu_mse}, {MLE_kappa_mse}, {MLE_lambda_mse}, {MLE_time_mean}")
        print("Method2(W1):")
        print(
            f"{W1method2_mu_mse}, {W1method2_kappa_mse}, "
            f"{W1method2_lambda_mse}, {W1method2_time_mean}"
        )
        df.index.name = "log10N"
        df.to_csv("./data/ss_vonmises_MSE/ex9_ss_vonmises_MSE.csv")
    df.index.name = "log10N"
    print(df)
    df.to_csv("./data/ss_vonmises_MSE/ex9_ss_vonmises_MSE.csv")


if __name__ == "__main__":
    _main()
