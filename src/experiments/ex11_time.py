from functools import partial
import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..calc_semidiscrete_W_dist import method1, method2
from ..distributions import vonmises, wrapedcauchy

# for von mises distribution
vM_bounds = ((-np.pi, np.pi), (3, 7))

def vM_W1_cost_func2(x, bin_num, data_cumsum_hist):
    mu, kappa = x
    dist_cumsum_hist = vonmises.cumsum_hist(mu, kappa, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def vM_est_W1_method2(given_data):
    """Calc W1-estimator using method2

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = vonmises.cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        vM_W1_cost_func2, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    return optimize.minimize(
        cost_func,
        (0, 4.5),
        bounds=vM_bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )

def vM_Wp_cost_func3(x, given_data_normed_sorted, p: int):
    sample = vonmises.fast_quantile_sampling(x[0], x[1], len(given_data_normed_sorted))
    sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
    sample = np.sort(sample)
    return method1.method1(given_data_normed_sorted, sample, p=p, sorted=True)


def vM_est_W2_method3(given_data):
    """Calc W2-estimator using method3

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(
        vM_Wp_cost_func3, given_data_normed_sorted=given_data_norm_sorted, p=2
    )
    return optimize.minimize(
        cost_func,
        (0, 4.5),
        bounds=vM_bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


# for wrapped cauchy distribution
WC_bounds = ((-np.pi, np.pi), (0.01, 0.99))
def WC_W2_cost_func3(x, given_data_normed_sorted):
    sample = wrapedcauchy.quantile_sampling(
        x[0], x[1], len(given_data_normed_sorted)
    ) / (2 * np.pi)
    return method1.method1(given_data_normed_sorted, sample, p=2, sorted=True)


def WC_est_W2_method3(given_data: npt.NDArray[np.float64]):
    """Calc W2-estimator using method3

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    given_data_norm = given_data / (2 * np.pi)
    given_data_norm_sorted = np.sort(given_data_norm)
    cost_func = partial(WC_W2_cost_func3, given_data_normed_sorted=given_data_norm_sorted)
    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=WC_bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def WC_W1_method2_cost_func(x, bin_num, data_cumsum_hist):
    mu, rho = x
    dist_cumsum_hist = wrapedcauchy.cumsum_hist(mu, rho, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def WC_est_W1_method2(given_data):
    """Calc W1-estimator using method2

    Args:
        given_data (np.ndarray): [0, 2*pi]のデータ
    """
    bin_num = len(given_data)
    data_cumsum_hist = wrapedcauchy.cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        WC_W1_method2_cost_func, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    return optimize.minimize(
        cost_func,
        (0, 0.5),
        bounds=WC_bounds,
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )


def main():
    Ns = [100, 1000, 10000, 100000]
    repeat_time = 10
    time_sum = 0

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

    # for von mises distribution
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
            MLE = vonmises.MLE(vonmises.T(sample), N)
            e_time = time.perf_counter()
            MLE_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = vM_est_W1_method2(sample)
            e_time = time.perf_counter()
            W1method2_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = vM_est_W2_method3(sample)
            e_time = time.perf_counter()
            W2method3_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = vonmises.type0_estimate(sample, gamma=0.5)
            e_time = time.perf_counter()
            type0_gamma05_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = vonmises.type1_estimate(sample, beta=0.5)
            e_time = time.perf_counter()
            type1_beta05_time[i] = e_time - s_time


        df.loc[N, "vM_MLE_time"] = np.mean(MLE_time)
        df.loc[N, "vM_W1(method2)_time"] = np.mean(W1method2_time)
        df.loc[N, "vM_W2(method3)_time"] = np.mean(W2method3_time)
        df.loc[N, "vM_type0_gamma05_time"] = np.mean(type0_gamma05_time)
        df.loc[N, "vM_type1_beta05_time"] = np.mean(type1_beta05_time)

    # for wrapped cauchy distribution
    true_rho = 0.75
    MLE_time = np.zeros(repeat_time)
    W1method2_time = np.zeros(repeat_time)
    W2method3_time = np.zeros(repeat_time)

    for N in Ns:
        for i in range(repeat_time):
            sample = stats.wrapcauchy(loc=true_mu, c=true_rho).rvs(N)
            sample = np.remainder(sample, 2 * np.pi)

            s_time = time.perf_counter()
            MLE = wrapedcauchy.MLE_Kent(sample)
            e_time = time.perf_counter()
            MLE_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = WC_est_W1_method2(sample)
            e_time = time.perf_counter()
            W1method2_time[i] = e_time - s_time

            s_time = time.perf_counter()
            est = WC_est_W2_method3(sample)
            e_time = time.perf_counter()
            W2method3_time[i] = e_time - s_time

        df.loc[N, "WC_MLE_time"] = np.mean(MLE_time)
        df.loc[N, "WC_W1(method2)_time"] = np.mean(W1method2_time)
        df.loc[N, "WC_W2(method3)_time"] = np.mean(W2method3_time)

    print(df)
    df.to_csv("./data/csv_data/time_comparison.csv")



if __name__ == "__main__":
    main() 