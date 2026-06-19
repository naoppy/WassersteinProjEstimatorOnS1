"""
データの生成分布がフォンミーゼス分布のときに、巻き込みコーシー分布でフィッティングした際に平均と円周分散がどうなるかを調べる。
MLE, W1, W2で比較。
つまり、model-misspecificationの意味でのロバスト性を調べる。
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from scipy import optimize
from tqdm import tqdm

from src.distributions import wrappedcauchy
from src.method.wasserstein import circular_wasserstein_from_samples
from src.utils.dist_utils import calculate_distances_vonmises_wrappedcauchy


def W2_quantile_sampling_downsampled(
    given_data: np.ndarray,
    x0: Optional[np.ndarray] = None,
    M: int = 1000,
) -> optimize.OptimizeResult:
    """W2 quantile sampling with downsampled empirical quantiles for speedup."""
    given_data = np.remainder(given_data, 2 * np.pi)
    given_data_sorted = np.sort(given_data)

    # Downsample using equally-spaced indices
    indices = np.linspace(0, len(given_data_sorted) - 1, M).astype(np.intp)
    sample_downsampled = given_data_sorted[indices]
    sample_downsampled_norm = sample_downsampled / (2 * np.pi)

    def cost_func(x):
        def ppf_func(q):
            return wrappedcauchy.wrapcauchy_ppf_analytical(q, x[1], loc=x[0])

        # Generate model sample at M points
        q_grid = np.linspace(1e-9, 1.0 - 1e-9, M)
        model_sample = ppf_func(q_grid) / (2 * np.pi)
        return circular_wasserstein_from_samples(
            sample_downsampled_norm, model_sample, p=2, sorted=True
        )

    if x0 is None:
        raise ValueError("x0 is required for local minimization")
    return optimize.minimize(
        cost_func,
        x0,
        bounds=wrappedcauchy.bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


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

    # W2 method3 (quantile sampling - downsampled)
    s_time = time.perf_counter()
    est = W2_quantile_sampling_downsampled(sample, x0=MLE)
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
        "W2method3_mu": W2method3_mu,
        "W2method3_kl": w2m3_kl,
        "W2method3_w1": w2m3_w1,
        "W2method3_w2": w2m3_w2,
        "W2method3_time": W2method3_time,
    }


def main():
    # true distribution is von Mises
    true_mu = 0.3
    kappas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    N = 100000
    try_num = 100  # 実験実行用のデフォルト試行回数

    print(f"true mu={true_mu}, N={N}")
    print("(mu_mse, KL, W1, W2, W1_sq, W2_sq, time)")

    df = pd.DataFrame(
        index=kappas,
        columns=[
            "MLE_mu_mse",
            "MLE_KL",
            "MLE_W1",
            "MLE_W2",
            "MLE_W1_sq",
            "MLE_W2_sq",
            "W1(method2)_mu_mse",
            "W1(method2)_KL",
            "W1(method2)_W1",
            "W1(method2)_W2",
            "W1(method2)_W1_sq",
            "W1(method2)_W2_sq",
            "W2(method3)_mu_mse",
            "W2(method3)_KL",
            "W2(method3)_W1",
            "W2(method3)_W2",
            "W2(method3)_W1_sq",
            "W2(method3)_W2_sq",
        ],
    )

    for _j, true_kappa in enumerate(kappas):
        print(f"kappa={true_kappa}")

        # MSEをとるための試行回数
        result = Parallel(n_jobs=-1)(
            delayed(run_once)(i, true_mu, true_kappa, N)
            for i in tqdm(range(try_num), desc=f"kappa={true_kappa}")
        )
        df_trial = pd.DataFrame(result)

        # mu_mse (periodic wrap-around)
        MLE_mu_mse = np.mean(
            (np.remainder(df_trial["MLE_mu"] - true_mu + np.pi, 2 * np.pi) - np.pi) ** 2
        )
        W1method2_mu_mse = np.mean(
            (
                np.remainder(df_trial["W1method2_mu"] - true_mu + np.pi, 2 * np.pi)
                - np.pi
            )
            ** 2
        )
        W2method3_mu_mse = np.mean(
            (
                np.remainder(df_trial["W2method3_mu"] - true_mu + np.pi, 2 * np.pi)
                - np.pi
            )
            ** 2
        )

        # Means of distances
        MLE_kl_mean = df_trial["MLE_kl"].mean()
        MLE_w1_mean = df_trial["MLE_w1"].mean()
        MLE_w2_mean = df_trial["MLE_w2"].mean()

        W1method2_kl_mean = df_trial["W1method2_kl"].mean()
        W1method2_w1_mean = df_trial["W1method2_w1"].mean()
        W1method2_w2_mean = df_trial["W1method2_w2"].mean()

        W2method3_kl_mean = df_trial["W2method3_kl"].mean()
        W2method3_w1_mean = df_trial["W2method3_w1"].mean()
        W2method3_w2_mean = df_trial["W2method3_w2"].mean()

        # Means of squared distances
        MLE_w1_sq_mean = (df_trial["MLE_w1"] ** 2).mean()
        MLE_w2_sq_mean = (df_trial["MLE_w2"] ** 2).mean()

        W1method2_w1_sq_mean = (df_trial["W1method2_w1"] ** 2).mean()
        W1method2_w2_sq_mean = (df_trial["W1method2_w2"] ** 2).mean()

        W2method3_w1_sq_mean = (df_trial["W2method3_w1"] ** 2).mean()
        W2method3_w2_sq_mean = (df_trial["W2method3_w2"] ** 2).mean()

        df.loc[true_kappa] = [
            MLE_mu_mse,
            MLE_kl_mean,
            MLE_w1_mean,
            MLE_w2_mean,
            MLE_w1_sq_mean,
            MLE_w2_sq_mean,
            W1method2_mu_mse,
            W1method2_kl_mean,
            W1method2_w1_mean,
            W1method2_w2_mean,
            W1method2_w1_sq_mean,
            W1method2_w2_sq_mean,
            W2method3_mu_mse,
            W2method3_kl_mean,
            W2method3_w1_mean,
            W2method3_w2_mean,
            W2method3_w1_sq_mean,
            W2method3_w2_sq_mean,
        ]

        print(
            f"MLE: mu_mse={MLE_mu_mse:.2e}, KL={MLE_kl_mean:.4f}, "
            f"W1={MLE_w1_mean:.4f}, W2={MLE_w2_mean:.4f}"
        )
        print(
            f"W1 eq: mu_mse={W1method2_mu_mse:.2e}, KL={W1method2_kl_mean:.4f}, "
            f"W1={W1method2_w1_mean:.4f}, W2={W1method2_w2_mean:.4f}"
        )
        print(
            f"W2 qu: mu_mse={W2method3_mu_mse:.2e}, KL={W2method3_kl_mean:.4f}, "
            f"W1={W2method3_w1_mean:.4f}, W2={W2method3_w2_mean:.4f}"
        )

    df.index.name = "kappa"
    print(df)
    df.to_csv("./data/model_misspecification/ex10_vM_WC.csv")


if __name__ == "__main__":
    main()
