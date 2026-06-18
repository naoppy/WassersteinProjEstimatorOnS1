"""
データの生成分布がフォンミーゼス分布のときに、巻き込みコーシー分布でフィッティングした際に平均と円周分散がどうなるかを調べる。
MLE, W1, W2で比較。
つまり、model-misspecificationの意味でのロバスト性を調べる。
"""

import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from tqdm import tqdm

from src.distributions import wrappedcauchy
from src.utils.dist_utils import calculate_distances_vonmises_wrappedcauchy


def run_once(i, true_mu, true_kappa, N: int) -> dict:
    # データはフォンミーゼス分布、モデルは巻き込みコーシー分布
    sample = stats.vonmises(loc=true_mu, kappa=true_kappa).rvs(N)
    sample = np.remainder(sample, 2 * np.pi)

    # MLE
    s_time = time.perf_counter()
    MLE = wrappedcauchy.MLE_Kent(sample)
    e_time = time.perf_counter()
    MLE_mu = MLE[0]
    MLE_rho = MLE[1]
    MLE_time = e_time - s_time

    mle_kl, _, mle_w1, mle_w2 = calculate_distances_vonmises_wrappedcauchy(
        true_mu, true_kappa, MLE_mu, MLE_rho
    )

    # W1 method2 (equal division)
    s_time = time.perf_counter()
    est = wrappedcauchy.W1_equal_div(sample, x0=MLE)
    e_time = time.perf_counter()
    W1method2_mu = est.x[0]
    W1method2_rho = est.x[1]
    W1method2_time = e_time - s_time

    w1m2_kl, _, w1m2_w1, w1m2_w2 = calculate_distances_vonmises_wrappedcauchy(
        true_mu, true_kappa, W1method2_mu, W1method2_rho
    )

    # W1 method3 (quantile sampling)
    s_time = time.perf_counter()
    est = wrappedcauchy.W1_quantile_sampling(sample, x0=MLE)
    e_time = time.perf_counter()
    W1method3_mu = est.x[0]
    W1method3_rho = est.x[1]
    W1method3_time = e_time - s_time

    w1m3_kl, _, w1m3_w1, w1m3_w2 = calculate_distances_vonmises_wrappedcauchy(
        true_mu, true_kappa, W1method3_mu, W1method3_rho
    )

    # W2 method3 (quantile sampling)
    s_time = time.perf_counter()
    est = wrappedcauchy.W2_quantile_sampling(sample, x0=MLE)
    e_time = time.perf_counter()
    W2method3_mu = est.x[0]
    W2method3_rho = est.x[1]
    W2method3_time = e_time - s_time

    w2m3_kl, _, w2m3_w1, w2m3_w2 = calculate_distances_vonmises_wrappedcauchy(
        true_mu, true_kappa, W2method3_mu, W2method3_rho
    )

    return {
        "MLE_mu": MLE_mu,
        "MLE_kl": mle_kl,
        "MLE_w1": mle_w1,
        "MLE_w2": mle_w2,
        "MLE_time": MLE_time,
        "W1method2_mu": W1method2_mu,
        "W1method2_kl": w1m2_kl,
        "W1method2_w1": w1m2_w1,
        "W1method2_w2": w1m2_w2,
        "W1method2_time": W1method2_time,
        "W1method3_mu": W1method3_mu,
        "W1method3_kl": w1m3_kl,
        "W1method3_w1": w1m3_w1,
        "W1method3_w2": w1m3_w2,
        "W1method3_time": W1method3_time,
        "W2method3_mu": W2method3_mu,
        "W2method3_kl": w2m3_kl,
        "W2method3_w1": w2m3_w1,
        "W2method3_w2": w2m3_w2,
        "W2method3_time": W2method3_time,
    }


def main():
    # true distribution is von Mises
    true_mu = 0.3
    true_kappa = 2

    print(f"true mu={true_mu}, true kappa={true_kappa}")
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

        # MSEをとるための試行回数
        result = Parallel(n_jobs=-1)(
            delayed(run_once)(i, true_mu, true_kappa, N)
            for i in tqdm(range(try_num), desc=f"N={N}")
        )
        df_trial = pd.DataFrame(result)

        # MSEを計算する
        MLE_mu_mse = np.mean(
            (np.remainder(df_trial["MLE_mu"] - true_mu + np.pi, 2 * np.pi) - np.pi) ** 2
        )
        MLE_kl_mean = df_trial["MLE_kl"].mean()
        MLE_w1_mean = df_trial["MLE_w1"].mean()
        MLE_w2_mean = df_trial["MLE_w2"].mean()
        MLE_time_mean = df_trial["MLE_time"].mean()

        W1method2_mu_mse = np.mean(
            (
                np.remainder(df_trial["W1method2_mu"] - true_mu + np.pi, 2 * np.pi)
                - np.pi
            )
            ** 2
        )
        W1method2_kl_mean = df_trial["W1method2_kl"].mean()
        W1method2_w1_mean = df_trial["W1method2_w1"].mean()
        W1method2_w2_mean = df_trial["W1method2_w2"].mean()
        W1method2_time_mean = df_trial["W1method2_time"].mean()

        W1method3_mu_mse = np.mean(
            (
                np.remainder(df_trial["W1method3_mu"] - true_mu + np.pi, 2 * np.pi)
                - np.pi
            )
            ** 2
        )
        W1method3_kl_mean = df_trial["W1method3_kl"].mean()
        W1method3_w1_mean = df_trial["W1method3_w1"].mean()
        W1method3_w2_mean = df_trial["W1method3_w2"].mean()
        W1method3_time_mean = df_trial["W1method3_time"].mean()

        W2method3_mu_mse = np.mean(
            (
                np.remainder(df_trial["W2method3_mu"] - true_mu + np.pi, 2 * np.pi)
                - np.pi
            )
            ** 2
        )
        W2method3_kl_mean = df_trial["W2method3_kl"].mean()
        W2method3_w1_mean = df_trial["W2method3_w1"].mean()
        W2method3_w2_mean = df_trial["W2method3_w2"].mean()
        W2method3_time_mean = df_trial["W2method3_time"].mean()

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

    df.index.name = "log10N"
    print(df)
    df.to_csv("./data/csv_data/model_misspecification/ex10_vM_WC.csv")


if __name__ == "__main__":
    main()
