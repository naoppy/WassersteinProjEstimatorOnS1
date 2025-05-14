"""
Vonmisesのcdf, ppfが異常に遅いので、速度を測定してみる。

わかったこと
cdf(x_list) は早いのだが、cdf(x) for x in x_list は遅い。
なるべく一気にcdfを計算したほうが良い。
ppfはcdfを二分探索のたびに実行しているので遅い。
"""


import time

import numpy as np
from numpy import typing as npt
from scipy.stats import vonmises


def my_ppf(x: npt.NDArray[np.float64], mu, kappa):
    shape = x.shape
    x = x.reshape((-1,))
    assert np.all((0 <= x) & (x <= 1))
    dist = vonmises(loc=mu, kappa=kappa)
    eps = 1e-8  # 30bit精度程度  log2(eps/(2pi))
    ret_array = np.zeros(len(x))

    for i, xi in enumerate(x):
        if xi < eps:
            ret_array[i] = mu - np.pi
            continue
        if 1 - xi < eps:
            ret_array[i] = mu + np.pi
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


def my_ppf2(x: npt.NDArray[np.float64], mu, kappa):
    shape = x.shape
    x = x.reshape((-1,))
    # assert np.all((0 <= x) & (x <= 1))
    dist = vonmises(loc=mu, kappa=kappa)
    eps = 1e-8  # 30bit精度程度  log2(eps/(2pi))
    ret_array = np.zeros(len(x))

    for i, xi in enumerate(x):
        # binary search
        left = mu - np.pi
        right = mu + np.pi
        while right - left > eps:
            y = np.linspace(left, right, 32)
            z = dist.cdf(y)
            # TODO: この境界も二分探索する
            for j in range(1, len(z)):
                if z[j] <= xi:
                    left = y[j]
                else:
                    right = y[j]
                    break
        ret_array[i] = left
    return ret_array.reshape(shape)


def my_ppf3(x: npt.NDArray[np.float64], mu, kappa):
    """xがソートされた0から1の値のリストであることを前提に高速化する
    未完成です
    """
    shape = x.shape
    x = x.reshape((-1,))
    dist = vonmises(loc=mu, kappa=kappa)
    eps = 1e-8  # 30bit精度程度  log2(eps/(2pi))
    ret_array = np.zeros(len(x))

    lefts = [mu - np.pi]
    rights = [mu + np.pi]
    Ns = [len(x)]
    StartIdx = [0]
    for _ in range(6):  # 5ビット精度 * 6 = 30ビット精度
        next_lefts = []
        next_rights = []
        next_Ns = []

        for i in range(len(lefts)):
            left = lefts[i]
            right = rights[i]
            N = Ns[i]
            startidx = StartIdx[i]
            y = np.linspace(left, right, 32)
            z = dist.cdf(y)
            for j in range(1, len(z)):
                if z[j] <= x:
                    lefts.append(y[j])
                    rights.append(rights[-1])
                    Ns.append(N // 2)
                else:
                    lefts.append(lefts[-1])
                    rights.append(y[j])
                    Ns.append(N // 2)

        lefts = next_lefts
        rights = next_rights
        Ns = next_Ns
        # update startidx with cumulative sum

    return ret_array.reshape(shape)


def my_ppf4(x: npt.NDArray[np.float64], mu, kappa):
    """最初に一気にcdfを計算しておく
    26bit精度 (10^-7程度)
    """
    shape = x.shape
    x = x.reshape((-1,))
    dist = vonmises(loc=mu, kappa=kappa)

    # cdfを一気に計算しておく
    y, step = np.linspace(mu - np.pi, mu + np.pi, 1048576, retstep=True)  # 2^20
    z = dist.cdf(y)
    lefts = np.zeros(len(x))
    i = 0
    for j, xi in enumerate(x):
        while i < len(z) and z[i] < xi:
            i += 1
        if i == 0:
            lefts[j] = mu - np.pi
        else:
            lefts[j] = mu - np.pi + (i - 1) * step
    # now (lefts, lefts + step) に xi がある
    # これを二分探索する
    ret_array = np.zeros(len(x))
    for i, xi in enumerate(x):
        y2 = np.linspace(lefts[i], lefts[i] + step, 64)
        z2 = dist.cdf(y2)
        left2 = 0
        right2 = len(y2)
        while right2 - left2 > 1:
            mid = (left2 + right2) // 2
            if z2[mid] < xi:
                left2 = mid
            else:
                right2 = mid
        ret_array[i] = left2
    return ret_array.reshape(shape)


def my_ppf5(x: npt.NDArray[np.float64], mu, kappa):
    """最初に一気にcdfを計算しておく
    21bit精度 (10^-7程度)
    """
    shape = x.shape
    x = x.reshape((-1,))
    dist = vonmises(loc=mu, kappa=kappa)

    # cdfを一気に計算しておく
    y, step = np.linspace(mu - np.pi, mu + np.pi, 1048576, retstep=True)  # 2^20
    z = dist.cdf(y)
    lefts = np.zeros(len(x))
    i = 0
    for j, xi in enumerate(x):
        while i < len(z) and z[i] < xi:
            i += 1
        if i == 0:
            lefts[j] = mu - np.pi
        else:
            lefts[j] = mu - np.pi + (i - 1) * step
    # now (lefts, lefts + step) に xi がある
    return (lefts + step / 2).reshape(shape)


def _main():
    mu = 2 * np.pi
    kappa = 2
    N1 = int(1e5)
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

    # s_time = time.perf_counter()
    # for xi in x:
    #     val = dist.cdf(xi)
    # e_time = time.perf_counter()
    # print(f"cdf iterate time: {e_time - s_time}")

    s_time = time.perf_counter()
    ppf_val = dist.ppf(np.linspace(0.01, 0.99, N2))
    e_time = time.perf_counter()
    print(f"ppf time: {(e_time - s_time) * N1 / N2}")

    s_time = time.perf_counter()
    myppf_val = my_ppf(np.linspace(0.01, 0.99, N2), mu, kappa)
    e_time = time.perf_counter()
    print(f"myppf time: {(e_time - s_time) * N1 / N2}")

    s_time = time.perf_counter()
    myppf_val2 = my_ppf2(np.linspace(0.01, 0.99, N2), mu, kappa)
    e_time = time.perf_counter()
    print(f"myppf2 time: {(e_time - s_time) * N1 / N2}")

    s_time = time.perf_counter()
    myppf_val4 = my_ppf4(np.linspace(0.01, 0.99, N1), mu, kappa)
    e_time = time.perf_counter()
    print(f"myppf4 time: {e_time - s_time}")

    s_time = time.perf_counter()
    myppf_val5 = my_ppf5(np.linspace(0.01, 0.99, N1), mu, kappa)
    e_time = time.perf_counter()
    print(f"myppf5 time: {e_time - s_time}")


if __name__ == "__main__":
    _main()
