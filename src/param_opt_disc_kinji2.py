"""
The Statistics of Circular Optimal Transportにあるように、W1距離をLevelMedianを使って計算する。
連続と離散のW1距離を、離散と離散で近似するが、その差はO(1/N)で減少することが知られている。
"""

import time
from typing import Tuple

import numpy as np
import ot
from scipy import optimize
from scipy.stats import vonmises

import vonmises_MLE

def vonmises_cdf_disc(N: int, mu: float, kappa: float) -> np.ndarray:
    """[-pi, pi]をN等分して、その区間でのフォンミーゼス分布の確率密度を計算する。

    Args:
        N (int): 分割数
        mu (float): 平均パラメータ
        kappa (float): 精度パラメータ

    Returns:
        np.ndarray: F(i/N)の値
    """
    ret = [vonmises.cdf(x, loc=mu, kappa=kappa) for x in np.linspace(-np.pi, np.pi, N, endpoint=False)]
    ret = np.array(ret)
    return ret - ret[0] # normalize F(0) = 0

def estimate_param(sample_cdf: np.ndarray, N: int) -> Tuple[float, float]:
    pass

def main():
    N = 500
    mu1 = 0.3
    kappa1 = 2
    sample = vonmises(loc=mu1, kappa=kappa1).rvs(N)
    vonmises_MLE.plot_vonmises(sample, mu1, kappa1, N)

    # calc MLE
    time1 = time.perf_counter()
    T_data = vonmises_MLE.T(sample)
    mu_MLE, kappa_MLE = vonmises_MLE.MLE(T_data, N)
    time2 = time.perf_counter()
    print(f"MLE result: mu={mu_MLE}, kappa={kappa_MLE}, time={time2-time1}s")
    
    # calc W1-estimator
    time3 = time.perf_counter()
    hist_sample = np.histogram(sample, bins=N, range=(-np.pi, np.pi), density = True)[0]
    sample_cdf = np.cumsum(hist_sample) # ヒストグラムにしてからcdfを計算
    mu_est, kappa_est = estimate_param(sample_cdf, N)
    time4 = time.perf_counter()
    print(f"W1-estimator result: mu={mu_est}, kappa={kappa_est}, time={time4-time3}s")

if __name__ == "__main__":
    main()