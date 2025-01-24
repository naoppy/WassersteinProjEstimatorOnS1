"""
巻き込みコーシー分布版
"""

from functools import partial

import numpy as np
import scipy.stats as stats


# see section 3.2
def q(w, n, x) -> complex:
    return n / (np.sum(1 / (np.exp(1j * x) - 1 / w))) + 1 / w


def calc_MLE(x, N: int, iter_num=100) -> complex:
    """rho e^(j mu) で返す"""
    if len(x) != N:
        raise ValueError("The length of x must be equal to N")
    if N < 3:
        raise ValueError("N must be greater than or equal to 3")
    x = np.array(x)
    my_q = partial(q, n=N, x=x)

    def my_Q(theta):
        return my_q(my_q(theta))

    # 計算が面倒なので適当な初期値を設定
    v = 1 / 2 + 1j / 2
    for _ in range(iter_num):
        # print(v)
        v = my_Q(v)
    return v


def main():
    N = 100000
    mu = np.pi / 2
    rho = 0.2
    x = stats.wrapcauchy.rvs(c=rho, loc=mu, size=N)
    result = calc_MLE(x, N, iter_num=100)
    print(f"rho MLE: {np.abs(result)}")
    print(f"mu  MLE: {np.angle(result)}")

    # Example A.2
    # x = [-8, -5, -3, -1, 2, 7, 10]
    # print(calc_MLE(x, len(x), iter_num=10))

    # Example A.3
    # x = [-10065, -8678, -6, 0]
    # print(calc_MLE(x, len(x), iter_num=10000))


if __name__ == "__main__":
    main()
