import time

import numpy as np
import pandas as pd
from scipy import stats

from src.distributions import vonmises, wrappedcauchy


def main():
    Ns = [100, 1000, 10000, 100000]
    repeat_time = 10

    df = pd.DataFrame(
        index=Ns,
        columns=[
            "vM_MLE_time",
            "vM_W1(method2)_time",
            "vM_W2(method3)_time",
            "vM_type0_gamma05_time",
            "vM_type1_beta05_time",
            "WC_MLE_time",
            "WC_W1(method2)_time",
            "WC_W2(method3)_time",
        ],
    )

    # for von Mises distribution
    true_mu = 0.3
    true_kappa = 5
    for N in Ns:
        MLE_time = np.zeros(repeat_time)
        W1method2_time = np.zeros(repeat_time)
        W2method3_time = np.zeros(repeat_time)
        type0_gamma05_time = np.zeros(repeat_time)
        type1_beta05_time = np.zeros(repeat_time)

        for i in range(repeat_time):
            sample = stats.vonmises(loc=true_mu, kappa=true_kappa).rvs(N)
            sample = np.remainder(sample, 2 * np.pi)

            s_time = time.perf_counter()
            vM_MLE = vonmises.MLE_direct(sample)
            e_time = time.perf_counter()
            MLE_time[i] = e_time - s_time

            s_time = time.perf_counter()
            _ = vonmises.W1_equal_div(sample, x0=vM_MLE)
            e_time = time.perf_counter()
            W1method2_time[i] = e_time - s_time

            s_time = time.perf_counter()
            _ = vonmises.W2_quantile_sampling(sample, x0=vM_MLE)
            e_time = time.perf_counter()
            W2method3_time[i] = e_time - s_time

            s_time = time.perf_counter()
            _ = vonmises.type0_estimate(sample, gamma=0.5)
            e_time = time.perf_counter()
            type0_gamma05_time[i] = e_time - s_time

            s_time = time.perf_counter()
            _ = vonmises.type1_estimate(sample, beta=0.5)
            e_time = time.perf_counter()
            type1_beta05_time[i] = e_time - s_time

        df.loc[N, "vM_MLE_time"] = np.mean(MLE_time)
        df.loc[N, "vM_W1(method2)_time"] = np.mean(W1method2_time)
        df.loc[N, "vM_W2(method3)_time"] = np.mean(W2method3_time)
        df.loc[N, "vM_type0_gamma05_time"] = np.mean(type0_gamma05_time)
        df.loc[N, "vM_type1_beta05_time"] = np.mean(type1_beta05_time)

    # for wrapped Cauchy distribution
    true_rho = 0.75
    for N in Ns:
        MLE_time = np.zeros(repeat_time)
        W1method2_time = np.zeros(repeat_time)
        W2method3_time = np.zeros(repeat_time)

        for i in range(repeat_time):
            sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
            sample = np.remainder(sample, 2 * np.pi)

            s_time = time.perf_counter()
            wc_MLE = wrappedcauchy.MLE_Kent(sample)
            e_time = time.perf_counter()
            MLE_time[i] = e_time - s_time

            s_time = time.perf_counter()
            _ = wrappedcauchy.W1_equal_div(sample, x0=wc_MLE)
            e_time = time.perf_counter()
            W1method2_time[i] = e_time - s_time

            s_time = time.perf_counter()
            _ = wrappedcauchy.W2_quantile_sampling(sample, x0=wc_MLE)
            e_time = time.perf_counter()
            W2method3_time[i] = e_time - s_time

        df.loc[N, "WC_MLE_time"] = np.mean(MLE_time)
        df.loc[N, "WC_W1(method2)_time"] = np.mean(W1method2_time)
        df.loc[N, "WC_W2(method3)_time"] = np.mean(W2method3_time)

    print(df)
    df.to_csv("./data/csv_data/time_comparison.csv")


if __name__ == "__main__":
    main()
