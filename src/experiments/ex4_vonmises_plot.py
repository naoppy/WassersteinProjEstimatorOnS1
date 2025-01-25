"""実験4
フォンミーゼス分布のパラメータとワッサースタイン距離のグラフを描画して、関数の形を調べる

結果：単峰っぽい形が出てきた、勾配降下法とかと相性が良さそう。
とりあえず適当にパウエル法やネルダーミード法で動かしてもちゃんと真値近くに収束するので、グリッドサーチいらなそう。

"""

from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from ..calc_semidiscrete_W_dist import method1, method2
from ..plots import brute_heatmap
from ..vonmises import vonmises_cumsum_hist
from ..vonmises import vonmises_MLE as vonmises_MLE


def est_W1_method2(given_data) -> Tuple[float, float]:
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        Tuple[float, float]: [推定したmu、推定したkappa]
    """
    bin_num = len(given_data)
    data_cumsum_hist = vonmises_cumsum_hist.cumsum_hist_data(given_data, bin_num)

    def cost_func(x):
        mu, kappa = x
        dist_cumsum_hist = vonmises_cumsum_hist.cumsum_hist(mu, kappa, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def est_W1_method2_justopt(given_data) -> Tuple[float, float]:
    bin_num = len(given_data)
    data_cumsum_hist = vonmises_cumsum_hist.cumsum_hist_data(given_data, bin_num)

    def cost_func(x):
        mu, kappa = x
        dist_cumsum_hist = vonmises_cumsum_hist.cumsum_hist(mu, kappa, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.minimize(
        cost_func,
        (0, 1),
        bounds=((-np.pi, np.pi), (0, 10)),
        # for powell method
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
        # for Nelder-Mead method
        # method="Nelder-Mead",
        # options={"xatol": 1e-6, "fatol": 1e-6},
    )


def est_W2_method1(given_data):
    """Calc W2-estimator using method1"""
    given_data_norm = np.remainder(given_data, 2 * np.pi) / (2 * np.pi)

    def cost_func(x):
        sample = stats.vonmises(loc=x[0], kappa=x[1]).rvs(len(given_data))
        sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
        return method1.method1(given_data_norm, sample, p=2)

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        # ((-0.4, 0.6), (2, 4)),
        full_output=True,
        finish=None,
        Ns=100,
    )


def est_W2_method1_justopt(given_data):
    """Calc W2-estimator using method1"""
    given_data_norm = np.remainder(given_data, 2 * np.pi) / (2 * np.pi)

    def cost_func(x):
        sample = stats.vonmises(loc=x[0], kappa=x[1]).rvs(len(given_data))
        sample = np.remainder(sample, 2 * np.pi) / (2 * np.pi)
        return method1.method1(given_data_norm, sample, p=2)

    return optimize.minimize(
        cost_func,
        (0, 1),
        bounds=((-np.pi, np.pi), (0, 10)),
        method="powell",
        options={"xtol": 1e-9, "ftol": 1e-9},
    )


def main():
    N = 1000
    mu1 = 0.3
    kappa1 = 2

    print(f"N={N}, True parameter: mu={mu1}, kappa={kappa1}")

    sample = stats.vonmises(loc=mu1, kappa=kappa1).rvs(N)
    # vonmises_MLE.plot_vonmises(sample, mu1, kappa1, N)

    print(f"MLE: {vonmises_MLE.MLE(vonmises_MLE.T(sample), N)}")

    ret = est_W1_method2(sample)
    brute_heatmap.plot_heatmap(ret, ("mu", "kappa"))

    ret2 = est_W1_method2_justopt(sample)
    print(ret2)

    ret3 = est_W2_method1(sample)
    brute_heatmap.plot_heatmap(ret3, ("mu", "kappa"))

    ret4 = est_W2_method1_justopt(sample)
    print(ret4)


if __name__ == "__main__":
    main()
