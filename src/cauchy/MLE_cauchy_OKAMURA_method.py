"""
CHARACTERIZATIONS OF THE MAXIMUM LIKELIHOOD ESTIMATOR OF THE CAUCHY DISTRIBUTION
https://arxiv.org/abs/2104.06130
で提案されているコーシー分布に対する最尤推定法を実装する
指数的に収束する反復法で、MLEに必ず収束する
"""

from functools import partial

import numpy as np


# see (2.11)
def mebius_t(theta: complex, eta: complex) -> complex:
    return (eta - theta) / (eta - np.conj(theta))


# see (2.12)
def mebius_t_inv(theta: complex, eta: complex) -> complex:
    return (theta - np.conj(theta) * eta) / (1 - eta)


# see (2.9)
def q(theta, n, x) -> complex:
    return n / (np.sum(1 / (x - theta))) + theta


def Q(theta, n, x) -> complex:
    return q(q(theta, n, x), n, x)


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
