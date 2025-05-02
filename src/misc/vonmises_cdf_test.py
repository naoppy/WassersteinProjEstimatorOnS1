"""
Vonmisesのcdf, ppfが異常に遅いので、速度を測定してみる。
"""

import time

import numpy as np
from numpy import typing as npt
from scipy.stats import vonmises, vonmises_line


def my_ppf(x: npt.NDArray[np.float64], mu, kappa):
    shape = x.shape
    x = x.reshape((-1,))
    assert np.all((0 <= x) & (x <= 1))
    dist = vonmises(loc=mu, kappa=kappa)
    eps = 1e-8
    ret_array = np.zeros(len(x))
   
    for i, xi in enumerate(x):
        if xi < eps:
            ret_array[i] = mu - np.pi
            continue
        if 1 - xi < eps:
            ret_array[i] =  mu + np.pi
            continue
        # now xi in (eps, 1-eps)
        # binary search
        left = mu - np.pi
        right = mu + np.pi
        while right - left > eps:
            mid = (right + left) / 2
            val = dist.cdf(mid)
            if val < xi:
                left = mid
            else:
                right = mid
        ret_array[i] = left
    return ret_array.reshape(shape)


def _main():
    mu = 2 * np.pi
    kappa = 2
    N1 = int(1e6)
    N2 = int(1e3)
    dist = vonmises(loc=mu, kappa=kappa)

    s_time = time.perf_counter()
    sample = dist.rvs(N1)
    e_time = time.perf_counter()
    print(f"rvs time: {e_time - s_time}")

    x = np.linspace(-10, 10, N1)
    s_time = time.perf_counter()
    cdf_val = dist.cdf(x)
    e_time = time.perf_counter()
    print(f"cdf time: {e_time - s_time}")

    s_time = time.perf_counter()
    ppf_val = dist.ppf(np.linspace(0.01, 0.99, N2))
    e_time = time.perf_counter()
    print(f"ppf time: {(e_time - s_time) * N1 / N2}")

    s_time = time.perf_counter()
    myppf_val = my_ppf(np.linspace(0.01, 0.99, N2), mu, kappa)
    e_time = time.perf_counter()
    print(f"myppf time: {(e_time - s_time) * N1 / N2}")

    s_time = time.perf_counter()
    for xi in x:
        val = dist.cdf(xi)
    e_time = time.perf_counter()
    print(f"cdf iterate time: {e_time - s_time}")



if __name__ == "__main__":
    _main()