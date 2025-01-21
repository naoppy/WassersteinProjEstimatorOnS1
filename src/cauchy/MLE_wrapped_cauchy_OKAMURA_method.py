"""
巻き込みコーシー分布版
"""

from functools import partial

import numpy as np


def q(theta, n, x) -> complex:
    return n / (np.sum(1 / (x - theta))) + theta


def calc_MLE(x, N: int, iter_num=100) -> complex:
    """mu + i * gamma で返す"""
    if len(x) != N:
        raise ValueError("The length of x must be equal to N")
    if N < 3:
        raise ValueError("N must be greater than or equal to 3")
    x = np.array(x)
    my_q = partial(q, n=N, x=x)

    def my_Q(theta):
        return my_q(my_q(theta))

    # 本当は初期値は median(x) + i * IRQ(x) が望ましい
    # 計算が面倒なので適当な初期値を設定
    v = 1 + 1j
    for _ in range(iter_num):
        print(v)
        v = my_Q(v)
    return v


def main():
    N = 1000
    mu = 2.0
    gamma = 5.0
    x = np.random.standard_cauchy(N) * gamma + mu
    print(calc_MLE(x, N))

    # Example A.2
    # x = [-8, -5, -3, -1, 2, 7, 10]
    # print(calc_MLE(x, len(x), iter_num=10))

    # Example A.3
    # x = [-10065, -8678, -6, 0]
    # print(calc_MLE(x, len(x), iter_num=10000))


if __name__ == "__main__":
    main()
