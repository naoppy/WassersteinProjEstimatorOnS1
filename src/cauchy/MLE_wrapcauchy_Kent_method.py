"""
巻き込みコーシー分布の最尤推定をケントの方法で行う
Maximum Likelihood Estimation for Wrapped Cauchy Distribution, Kent and Tyler, 1988
"""

import numpy as np
import numpy.typing as npt
import scipy.stats as stats


def cossin(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(cos(x), sin(x))を返す

    Args:
        x (npt.NDArray[np.float64]): 0~2piの角度データ

    Returns:
        npt.NDArray[np.float64]: [[cos(x[0]), cos(x[1]), ...], [sin(x[0]), sin(x[1]), ...]]
    """
    return np.array([np.cos(x), np.sin(x)])


def calc_MLE(x, tol=1e-6, max_iter=10000, debug=False) -> npt.NDArray[np.float64]:
    N = len(x)
    x = np.array(x)  # (N,)
    y = cossin(x)  # (2, N)
    eta = np.array([0.5, 0.5])  # (2,) 適当なノルム1未満の初期点
    for i in range(max_iter):
        w = 1 / (1 - eta @ y)  # (N,)
        assert w.shape == (N,)
        eta_new = np.sum(w * y, axis=1) / np.sum(w)  # (2,)
        assert eta_new.shape == (2,)
        if np.linalg.norm(eta_new - eta) < tol:
            if debug:
                print(f"wrapcauchy kent MLE: Converged at {i}th iteration")
            eta = eta_new
            break
        eta = eta_new
    mu = np.arctan2(eta[1], eta[0])
    eta_norm_pow2 = eta @ eta
    rho = (1 - np.sqrt(1 - eta_norm_pow2)) / np.sqrt(eta_norm_pow2)
    return np.array([rho, mu])


def main():
    N = 100000
    mu = np.pi / 2
    rho = 0.2
    x = stats.wrapcauchy.rvs(c=rho, loc=mu, size=N)
    result = calc_MLE(x, debug=True, tol=1e-9)
    print(f"rho MLE by Kent: {result[0]}")
    print(f"mu  MLE by Kent: {result[1]}")


if __name__ == "__main__":
    main()
