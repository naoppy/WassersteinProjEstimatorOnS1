"""実験4
フォンミーゼス分布のパラメータとワッサースタイン距離のグラフを描画して、凸かどうか調べる
"""

import time
from typing import Tuple

import numpy as np
import scipy.stats as stats
from scipy import optimize

from ..calc_semidiscrete_W_dist import method2
from ..vonmises import vonmises_cumsum_hist
from ..vonmises import vonmises_MLE as vonmises_MLE


def estimate_param(given_data) -> Tuple[float, float]:
    """与えられたデータから、最適なパラメータを推定する

    Args:
        given_data (np.ndarray): サンプル1、[-pi, pi]のデータ

    Returns:
        Tuple[float, float]: [推定したmu、推定したkappa]
    """
    n = len(given_data)
    bin_num = len(given_data)
    data_hist = np.zeros(bin_num + 1)
    for x in given_data:
        data_hist[
            np.clip(int((x + np.pi) / (2 * np.pi) * bin_num) + 1, 1, bin_num)
        ] += 1
    data_cumsum_hist = np.cumsum(data_hist) / n
    assert abs(data_cumsum_hist[0] - 0.0) < 1e-7
    assert abs(data_cumsum_hist[-1] - 1.0) < 1e-7

    def cost_func(x):
        mu, kappa = x
        dist_cumsum_hist = vonmises_cumsum_hist.cumsum_hist(mu, kappa, bin_num)
        return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])

    return optimize.brute(
        cost_func,
        ((-np.pi, np.pi), (0, 10)),
        full_output=True,
        finish=None,
        Ns=10,
    )


def main():
    N = 10000
    mu1 = 0.3
    kappa1 = 2

    print(f"N={N}, True parameter: mu={mu1}, kappa={kappa1}")

    sample = stats.vonmises(loc=mu1, kappa=kappa1).rvs(N)
    # vonmises_MLE.plot_vonmises(sample, mu1, kappa1, N)

    time3 = time.perf_counter()
    ret = estimate_param(sample)
    print(ret)
    time4 = time.perf_counter()
    # print(
    #     f"Mehtod2 Estimation result: mu={mu_est}, kappa={kappa_est}, time={time4-time3}s"
    # )


if __name__ == "__main__":
    main()
