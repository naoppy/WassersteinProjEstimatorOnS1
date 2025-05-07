from functools import partial
from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.stats import wrapcauchy


def fisher_info_2x2(rho: float) -> npt.NDArray[np.float64]:
    """
    巻き込みコーシー分布のフィッシャー情報量を計算する
    """
    rho_p2 = rho * rho
    bunbo = (1 - rho_p2) ** 2
    return np.array(
        [
            [2 * rho_p2 / bunbo, 0],
            [0, 2 / bunbo],
        ]
    )


def fisher_mat_inv_diag(rho: float) -> List[float]:
    """
    巻き込みコーシー分布のフィッシャー情報量の逆行列の対角成分を計算する
    """
    rho_p2 = rho * rho
    bunbo = (1 - rho_p2) ** 2
    return [bunbo / (2 * rho_p2), bunbo / 2]


def wrapcauchy_true_pdf(
    x: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """[loc, loc+2*pi*scale] で定義されている pdf を R 全体に拡張"""
    norm_x = loc + np.remainder(x - loc, 2 * np.pi * scale)
    return wrapcauchy.pdf(norm_x, c, loc, scale)


def wrapcauchy_periodic_cdf(
    x: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """[loc, loc+2*pi*scale] で定義されている cdf を R 全体に拡張"""
    norm_x = loc + np.remainder(x - loc, 2 * np.pi * scale)
    return wrapcauchy.cdf(norm_x, c, loc, scale) + np.floor_divide(
        x - loc, 2 * np.pi * scale
    )


def cumsum_hist(mu: float, rho: float, bin_num) -> npt.NDArray[np.float64]:
    """[0, 2pi] を bin_num 等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    x = np.linspace(0, 2 * np.pi, bin_num + 1)
    y = wrapcauchy_periodic_cdf(x, rho, mu, 1) - wrapcauchy_periodic_cdf(0, rho, mu, 1)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def cumsum_hist_data(sample, bin_Num) -> npt.NDArray[np.float64]:
    """[0, 2pi] を bin_num 等分した区間でのデータのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    n = len(sample)
    data_hist = np.zeros(bin_Num + 1)
    for x in sample:
        data_hist[np.clip(int(x / (2 * np.pi) * bin_Num) + 1, 1, bin_Num)] += 1
    data_cumsum_hist = np.cumsum(data_hist) / n

    assert abs(data_cumsum_hist[0] - 0.0) < 1e-7
    assert abs(data_cumsum_hist[-1] - 1.0) < 1e-7
    return data_cumsum_hist


def quantile_sampling(
    mu: float, rho: float, sample_num: int
) -> npt.NDArray[np.float64]:
    """巻き込みコーシー分布から分位点サンプリングする
    ソート済みの点列を返す

    Args:
        mu (float): 分布のパラメータ
        rho (float): 分布のパラメータ
        sample_num (int): サンプルする数

    Returns:
        npt.NDArray[np.float64]: [0, 2pi] の範囲のサンプル。F^(-1)(i/D) (i=0, 1, ..., D)
    """
    x = np.linspace(0, 1, sample_num)
    assert len(x) == sample_num
    y = wrapcauchy.ppf(x, rho, loc=mu)
    y = np.remainder(y, 2 * np.pi)
    # sort
    i = 0
    if y[-1] <= y[0]:  # y2[i-1] <= y[i]
        i += 1
        while y[i - 1] <= y[i]:
            i += 1
    # 既にソート済みならi=0, 全て同じ値(pdfがデルタ関数)ならi=sample_num, それ以外ならiは変曲点の奥のidx
    y = np.roll(y, -i)
    assert np.all((0 <= y) & (y <= 2 * np.pi))
    assert np.all(y[i] <= y[i + 1] for i in range(sample_num - 1))
    return y


# see section 3.2
def _q(w, n, x) -> complex:
    return n / (np.sum(1 / (np.exp(1j * x) - 1 / w))) + 1 / w


def MLE_OKAMURA(x, N: int, iter_num=100) -> npt.NDArray[np.float64]:
    """[mu, rho] で返す

    CHARACTERIZATIONS OF THE MAXIMUM LIKELIHOOD ESTIMATOR OF THE CAUCHY DISTRIBUTION
    https://arxiv.org/abs/2104.06130
    で提案されているコーシー分布に対する最尤推定法を実装する
    指数的に収束する反復法で、MLEに必ず収束する

    Args:
        x (npt.NDArray[np.float64]): 0~2piの角度データ
        N (int): データ数
        iter_num (int, optional): 反復回数. Defaults to 100.

    Returns:
        npt.NDArray[np.float64]: [mu, rho]。muは [-pi, pi] の範囲。rhoは [0, 1] の範囲
    """
    if len(x) != N:
        raise ValueError("The length of x must be equal to N")
    if N < 3:
        raise ValueError("N must be greater than or equal to 3")
    x = np.array(x)
    my_q = partial(_q, n=N, x=x)

    def my_Q(theta):
        return my_q(my_q(theta))

    # 計算が面倒なので適当な初期値を設定
    v = 1 / 2 + 1j / 2
    for _ in range(iter_num):
        # print(v)
        v = my_Q(v)
    # v = rho e^(j mu) になっている
    return np.array([np.angle(v), np.abs(v)])


def _cossin(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(cos(x), sin(x))を返す

    Args:
        x (npt.NDArray[np.float64]): 0~2piの角度データ

    Returns:
        npt.NDArray[np.float64]: [[cos(x[0]), cos(x[1]), ...], [sin(x[0]), sin(x[1]), ...]]
    """
    return np.array([np.cos(x), np.sin(x)])


def MLE_Kent(x, tol=1e-6, max_iter=10000, debug=False) -> npt.NDArray[np.float64]:
    """
    巻き込みコーシー分布の最尤推定をケントの方法で行う
    Maximum Likelihood Estimation for Wrapped Cauchy Distribution, Kent and Tyler, 1988

    Args:
        x (npt.NDArray[np.float64]): 0~2piの角度データ
        tol (float, optional): 収束判定の閾値. Defaults to 1e-6.
        max_iter (int, optional): 最大反復回数. Defaults to 10000.
        debug (bool, optional): デバッグモード. Defaults to False.

    Returns:
        npt.NDArray[np.float64]: [mu_MLE, rho_MLE]。muは [-pi, pi] の範囲。rhoは [0, 1] の範囲
    """
    N = len(x)
    x = np.array(x)  # (N,)
    y = _cossin(x)  # (2, N)
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
    return np.array([mu, rho])


def _pdf_scale2pi(theta, mu, rho):
    return (1 - rho * rho) / (1 + rho * rho - 2 * rho * np.cos(theta - mu))


# 負の対数尤度関数の定義
def neg_log_likelihood(params, data):
    mu, rho = params
    pdf_vals = _pdf_scale2pi(data, mu, rho)
    # 小さな値を避けるためにクリッピング
    eps = 1e-10
    log_likelihood = np.sum(np.log(np.clip(pdf_vals, eps, None)))
    return -log_likelihood  # 最小化関数用にマイナスを返す


def MLE_direct_opt(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result = optimize.minimize(
        neg_log_likelihood,
        (0, 0.5),
        args=(x,),
        bounds=((-np.pi, np.pi), (0.01, 0.99)),
        method="powell",
        options={"xtol": 1e-6, "ftol": 1e-6},
    )
    return np.array([result.x[0], result.x[1]])


def main():
    mu = np.pi / 2  # circular mean
    rho = 0.7  # concentration
    N = 10000
    dist = wrapcauchy(loc=mu, c=rho)

    # calc Fisher info matrix
    print("Fisher info:")
    print(fisher_info_2x2(rho))

    sample = dist.rvs(N)
    print(f"min: {np.min(sample)}, max: {np.max(sample)}")  # [mu, mu + 2pi]

    sample2 = np.remainder(sample, 2 * np.pi)
    print(f"min: {np.min(sample2)}, max: {np.max(sample2)}")  # [0, 2pi]

    # calc MLE
    result = MLE_OKAMURA(sample, N, iter_num=100)
    print(f"mu  MLE: {result[0]}")
    print(f"rho MLE: {result[1]}")
    result2 = MLE_Kent(sample, debug=True, tol=1e-9)
    print(f"mu MLE by Kent: {result2[0]}")
    print(f"rho MLE by Kent: {result2[1]}")
    result3 = MLE_direct_opt(sample)
    print(f"mu MLE by direct: {result3[0]}")
    print(f"rho MLE by direct: {result3[1]}")

    sample2 = quantile_sampling(mu, rho, N)
    print(f"min: {np.min(sample2)}, max: {np.max(sample2)}")  # [0, 2pi]

    # plots
    # ライブラリのCDFはmuからmu+2piで定義されている。
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    plt.plot(x, wrapcauchy.cdf(x, rho, mu), label="cdf of scipy")
    plt.plot(x, wrapcauchy_true_pdf(x, rho, mu), label="periodic pdf")
    plt.plot(x, wrapcauchy_periodic_cdf(x, rho, mu), label="periodic cdf")
    plt.plot(
        x,
        wrapcauchy_periodic_cdf(x, rho, mu) - wrapcauchy_periodic_cdf(0, rho, mu),
        label="normalized cdf",
    )
    plt.legend()
    plt.show()

    # 普通のPPFは0でmu、1でmu+2piになる
    # つまり、普通のCDFの逆関数になるようになっている。
    # 0で0、1で2piになるようなPPFにすることもできるが、多くの場合する必要はない。
    x = np.linspace(0, 1, 1001)
    y = dist.ppf(x)
    print("min:", np.min(y), "max:", np.max(y))  # [mu, mu + 2pi]
    plt.plot(x, y, label="ppf")
    y2 = np.remainder(y, 2 * np.pi)
    plt.plot(x, y2, label="ppf mod 2pi")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
