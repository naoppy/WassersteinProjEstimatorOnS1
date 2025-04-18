from typing import Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import integrate, optimize, stats
from scipy.special import i0, i1, iv


def pdf(
    x: npt.NDArray[np.float64], mu: float, kappa: float, lambda_: float
) -> npt.NDArray[np.float64]:
    """Sine-skewed von Mises分布の確率密度関数を計算する

    Args:
        x (npt.NDArray[np.float64]): 確率密度関数を計算する点 in [-pi, pi]
        mu (float): 歪める前の分布の平均 in [-pi, pi]
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]

    Returns:
        npt.NDArray[np.float64]: 確率密度関数の値 in [0, 1]
    """
    return stats.vonmises.pdf(x, loc=mu, kappa=kappa) * (1 + lambda_ * np.sin(x - mu))


def cdf(
    x: npt.NDArray[np.float64], mu: float, kappa: float, lambda_: float
) -> npt.NDArray[np.float64]:
    """Sine-Skewed von Mises分布の累積分布関数を計算する
    cdf(mu-pi) = 0, cdf(mu+pi) = 1 となるように定義されている。

    Args:
        x (npt.NDArray[np.float64]): 累積分布関数を計算する点 in [-pi, pi]
        mu (float): 歪める前の分布の平均 in [-pi, pi]
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]

    Returns:
        npt.NDArray[np.float64]: 累積分布関数の値 in [0, 1]
    """
    dist = stats.vonmises(loc=mu, kappa=kappa)
    return dist.cdf(x, loc=mu, kappa=kappa) + lambda_ / (
        2 * np.pi * i0(kappa) * kappa
    ) * (np.exp(-kappa) - np.exp(kappa * np.cos(x - mu)))


def fisher_info_3x3(kappa: float, lambda_: float) -> npt.NDArray[np.float64]:
    """Sine-Skewed von Mises分布のフィッシャー情報行列を計算する。
    数値積分を使って近似的な値を計算することに注意。
    パラメータはmu, kappa, lambda の順。

    Args:
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]

    Returns:
        npt.NDArray[np.float64]: フィッシャー情報行列
    """
    i_mu_mu = kappa * i1(kappa) / i0(kappa) + integrate.quad(
        lambda x: np.exp(kappa * np.cos(x))
        * (lambda_ + np.sin(x))
        / (1 + lambda_ * np.sin(x)),
        -np.pi,
        np.pi,
    )[0] / (2 * np.pi * i0(kappa))
    i_kappa_kappa = 1 / 2 + iv(2, kappa) / i0(kappa) - (i1(kappa) / i0(kappa)) ** 2
    i_lambda_lambda = integrate.quad(
        lambda x: np.exp(kappa * np.cos(x))
        * np.sin(x) ** 2
        / (1 + lambda_ * np.sin(x)),
        -np.pi,
        np.pi,
    )[0] / (2 * np.pi * i0(kappa))
    i_mu_kappa = lambda_ / 2 * (1 - iv(2, kappa) / i0(kappa))
    i_kappa_lambda = 0
    i_mu_lambda = integrate.quad(
        lambda x: np.exp(kappa * np.cos(x)) * np.cos(x) / (1 + lambda_ * np.sin(x)),
        -np.pi,
        np.pi,
    )[0] / (2 * np.pi * i0(kappa))
    return np.array(
        [
            [i_mu_mu, i_mu_kappa, i_mu_lambda],
            [i_mu_kappa, i_kappa_kappa, i_kappa_lambda],
            [i_mu_lambda, i_kappa_lambda, i_lambda_lambda],
        ]
    )


def neg_log_likelihood(params, data) -> float:
    mu, kappa, lambda_ = params
    eps = 1e-10
    data = data - mu
    n = len(data)
    log_likelihood = (
        -n * np.log(np.maximum(eps, i0(kappa)))
        + kappa * np.sum(np.cos(data))
        + np.sum(np.log(np.maximum(eps, 1 + lambda_ * np.sin(data))))
    )
    return -log_likelihood


def MLE_direct_opt(x: npt.NDArray[np.float64]) -> Tuple[float, float, float]:
    """SS-von MisesのMLEでのパラメータ推定を行う

    Args:
        x (npt.NDArray[np.float64]): データ in [-pi, pi]

    Returns:
        Tuple[float, float, float]: (mu, kappa, lambda) in [-pi, pi]x[0, inf]x[-1, 1]
    """
    result = optimize.differential_evolution(
        neg_log_likelihood,
        args=(x,),
        bounds=((-np.pi, np.pi), (0.01, 4), (-1, 1)),
    )
    print(result)
    return result.x


def rejection_sampling(
    n: int, mu: float, kappa: float, lambda_: float, debug: bool = False
) -> npt.NDArray[np.float64]:
    """棄却サンプリングによってSine-Skewed von Misesからサンプリングする
    提案分布としては 2倍のフォンミーゼス分布を用いる。

    Args:
        n (int): サンプル数
        mu (float): 歪める前の分布の平均 in [-pi, pi]
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]
        debug (bool, optional): 棄却率を出力するか

    Returns:
        npt.NDArray[np.float64]: サンプル in [-pi, pi]
    """
    rng = np.random.default_rng()

    cnt = 0
    try_num = 0
    ret = np.zeros(n)
    while cnt < n:
        # 一気に 2 * 必要数サンプルする (棄却率が50%のため)
        xs = stats.vonmises(loc=mu, kappa=kappa).rvs(2 * (n - cnt))
        for x in xs:
            if cnt >= n:
                break
            try_num += 1
            if 2 * rng.random() < 1 + lambda_ * np.sin(x - mu):
                # accept
                ret[cnt] = x
                cnt += 1

    if debug:
        print(f"accept rate: {n / try_num}")
    return ret

def cumsum_hist(mu: float, kappa: float, lambda_: float, bin_num: int) -> npt.NDArray[np.float64]:
    """[-pi, pi] の間を bin_num (=D) 等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    F(0)=0, F(1)=1を満たすソート済み列を返す
    """
    x = np.linspace(-np.pi, np.pi, bin_num + 1)
    y = cdf(x, mu, kappa, lambda_) - cdf(-np.pi, mu, kappa, lambda_)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7, print(f"{mu}, {kappa}, {lambda_}, {y}")
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def cumsum_hist_data(data: npt.NDArray[np.float64], bin_num: int) -> npt.NDArray[np.float64]:
    pass

def _main():
    n = 100000
    kappa = 1
    lambda_ = 0.7
    mu = 0
    sample = rejection_sampling(n, mu, kappa, lambda_, debug=True)
    print(f"min: {np.min(sample)}, max: {np.max(sample)}")  # [-pi, pi]

    fig = plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")
    x = np.linspace(-np.pi, np.pi, 1000)
    ss_vonmises_pdf = pdf(x, mu, kappa, lambda_)
    ticks = [0, 0.15, 0.3]

    left.plot(x, ss_vonmises_pdf)
    left.set_yticks(ticks)
    number_of_bins = int(np.sqrt(n))
    left.hist(sample, density=True, bins=number_of_bins)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x, ss_vonmises_pdf, label="PDF")
    right.set_yticks(ticks)
    right.hist(sample, density=True, bins=number_of_bins, label="Histogram")
    right.set_title("Polar plot")
    right.legend(bbox_to_anchor=(0.15, 1.06))

    # param estimation
    est_param = MLE_direct_opt(sample)
    print(est_param)
    ss_vonmises_est_pdf = pdf(x, est_param[0], est_param[1], est_param[2])
    left.plot(x, ss_vonmises_est_pdf)
    right.plot(x, ss_vonmises_est_pdf)

    plt.show()

    mat = fisher_info_3x3(kappa, lambda_)
    print("Fisher info:")
    print(mat)


if __name__ == "__main__":
    _main()
