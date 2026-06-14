from functools import partial
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy import integrate, optimize, stats
from scipy.special import i0, i1, iv, ive

from ..calc_semidiscrete_W_dist import method2
from ..misc.circular_utils import (
    cumsum_hist_data,
    to_2pi_range,
)


def _bessel_ratio(v: int, kappa: float) -> float:
    """I_v(kappa) / I_0(kappa) を安全に計算する。
    オーバーフロー対策として、kappa >= 600 の場合は
    指数スケーリングされた ive を使用する。
    """
    if kappa < 600:
        if v == 0:
            return 1.0
        elif v == 1:
            return i1(kappa) / i0(kappa)
        else:
            return iv(v, kappa) / i0(kappa)
    else:
        return ive(v, kappa) / ive(0, kappa)


def sine_skewed_vonmises_pdf_analytical(
    x: npt.NDArray[np.float64], mu: float, kappa: float, lambda_: float
) -> npt.NDArray[np.float64]:
    """Sine-skewed von Mises分布の確率密度関数を計算する。

    Args:
        x (npt.NDArray[np.float64]): 確率密度関数を計算する点 in [0, 2pi]
        mu (float): 歪める前の分布の平均 in [0, 2pi]
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]

    Returns:
        npt.NDArray[np.float64]: 確率密度関数の値 in [0, 1]
    """
    x = to_2pi_range(x)
    mu = to_2pi_range(mu)
    return stats.vonmises.pdf(x, loc=mu, kappa=kappa) * (1 + lambda_ * np.sin(x - mu))


def sine_skewed_vonmises_periodic_cdf_analytical(
    x: npt.NDArray[np.float64], mu: float, kappa: float, lambda_: float
) -> npt.NDArray[np.float64]:
    """Sine-Skewed von Mises分布の累積分布関数を計算する。
    cdf(0) = 0, cdf(2*pi) = 1 となるように定義。

    Args:
        x (npt.NDArray[np.float64]): 累積分布関数を計算する点 in [0, 2pi]
        mu (float): 歪める前の分布の平均 in [0, 2pi]
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]

    Returns:
        npt.NDArray[np.float64]: 累積分布関数の値 in [0, 1]
    """
    x = to_2pi_range(x)
    mu = to_2pi_range(mu)

    def cdf_raw(val):
        if kappa < 600:
            return stats.vonmises.cdf(val, loc=mu, kappa=kappa) + lambda_ / (
                2 * np.pi * i0(kappa) * kappa
            ) * (np.exp(-kappa) - np.exp(kappa * np.cos(val - mu)))
        else:
            return stats.vonmises.cdf(val, loc=mu, kappa=kappa) + lambda_ / (
                2 * np.pi * ive(0, kappa) * kappa
            ) * (np.exp(-2 * kappa) - np.exp(kappa * (np.cos(val - mu) - 1)))

    return cdf_raw(x) - cdf_raw(0)


pdf = sine_skewed_vonmises_pdf_analytical
cdf = sine_skewed_vonmises_periodic_cdf_analytical


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
    r1 = _bessel_ratio(1, kappa)
    r2 = _bessel_ratio(2, kappa)

    if kappa < 600:
        scale_factor = 2 * np.pi * i0(kappa)
        i_mu_mu = (
            kappa * r1
            + lambda_
            * integrate.quad(
                lambda x: np.exp(kappa * np.cos(x))
                * (lambda_ + np.sin(x))
                / (1 + lambda_ * np.sin(x)),
                -np.pi,
                np.pi,
                points=[0.0],
            )[0]
            / scale_factor
        )
        i_lambda_lambda = (
            integrate.quad(
                lambda x: np.exp(kappa * np.cos(x))
                * (np.sin(x) ** 2)
                / (1 + lambda_ * np.sin(x)),
                -np.pi,
                np.pi,
                points=[0.0],
            )[0]
            / scale_factor
        )
        i_mu_lambda = (
            integrate.quad(
                lambda x: np.exp(kappa * np.cos(x))
                * np.cos(x)
                / (1 + lambda_ * np.sin(x)),
                -np.pi,
                np.pi,
                points=[0.0],
            )[0]
            / scale_factor
        )
    else:
        scale_factor_ive = 2 * np.pi * ive(0, kappa)
        i_mu_mu = (
            kappa * r1
            + lambda_
            * integrate.quad(
                lambda x: np.exp(kappa * (np.cos(x) - 1))
                * (lambda_ + np.sin(x))
                / (1 + lambda_ * np.sin(x)),
                -np.pi,
                np.pi,
                points=[0.0],
            )[0]
            / scale_factor_ive
        )
        i_lambda_lambda = (
            integrate.quad(
                lambda x: np.exp(kappa * (np.cos(x) - 1))
                * (np.sin(x) ** 2)
                / (1 + lambda_ * np.sin(x)),
                -np.pi,
                np.pi,
                points=[0.0],
            )[0]
            / scale_factor_ive
        )
        i_mu_lambda = (
            integrate.quad(
                lambda x: np.exp(kappa * (np.cos(x) - 1))
                * np.cos(x)
                / (1 + lambda_ * np.sin(x)),
                -np.pi,
                np.pi,
                points=[0.0],
            )[0]
            / scale_factor_ive
        )

    i_kappa_kappa = (1 + r2) / 2 - r1**2
    i_mu_kappa = lambda_ * (r2 - 1) / 2
    i_kappa_lambda = 0

    return np.array(
        [
            [i_mu_mu, i_mu_kappa, i_mu_lambda],
            [i_mu_kappa, i_kappa_kappa, i_kappa_lambda],
            [i_mu_lambda, i_kappa_lambda, i_lambda_lambda],
        ]
    )


def fisher_mat_inv_diag(kappa: float, lambda_: float) -> List[float]:
    mat = fisher_info_3x3(kappa, lambda_)
    mat_inv = np.linalg.inv(mat)
    return [mat_inv[0][0], mat_inv[1][1], mat_inv[2][2]]


def neg_log_likelihood(params, data) -> float:
    mu, kappa, lambda_ = params
    eps = 1e-10
    data = data - mu
    n = len(data)

    if kappa < 600:
        log_i0 = np.log(np.maximum(eps, i0(kappa)))
    else:
        log_i0 = np.log(np.maximum(eps, ive(0, kappa))) + kappa

    log_likelihood = (
        -n * log_i0
        + kappa * np.sum(np.cos(data))
        + np.sum(np.log(np.maximum(eps, 1 + lambda_ * np.sin(data))))
    )
    return -log_likelihood


def MLE_direct(
    x: npt.NDArray[np.float64],
    bounds=((0, 2 * np.pi), (0.01, 10.0), (-1.0, 1.0)),
    tol: float = 0.001,
    debug: bool = False,
) -> Tuple[float, float, float]:
    """SS-von MisesのMLEでのパラメータ推定を行う。

    Args:
        x (npt.NDArray[np.float64]): データ in [0, 2pi]
        bounds (tuple, optional): パラメータの探索範囲。
            デフォルトは ((0, 2*pi), (0.01, 10.0), (-1.0, 1.0))
        tol (float, optional): 最適化の収束判定閾値
        debug (bool, optional): 最適化の途中経過を出力するかどうか

    Returns:
        Tuple[float, float, float]: 推定値 (mu, kappa, lambda_) の順
    """
    x = to_2pi_range(x)
    result = optimize.differential_evolution(
        neg_log_likelihood,
        tol=tol,
        args=(x,),
        bounds=bounds,
    )
    if debug:
        print(result)
    return tuple(result.x)


MLE_direct_opt = MLE_direct


def rejection_sampling(
    n: int, mu: float, kappa: float, lambda_: float, debug: bool = False
) -> npt.NDArray[np.float64]:
    """棄却サンプリングによってSine-Skewed von Misesからサンプリングする。
    提案分布としては 2倍のフォンミーゼス分布を用いる。

    Args:
        n (int): サンプル数
        mu (float): 歪める前の分布の平均 in [0, 2pi]
        kappa (float): 分布のパラメータ (>0)
        lambda_ (float): 摂動項のパラメータ in [-1, 1]
        debug (bool, optional): 棄却率を出力するか

    Returns:
        npt.NDArray[np.float64]: [0, 2pi] の範囲のサンプル配列
    """
    rng = np.random.default_rng()
    mu = to_2pi_range(mu)

    cnt = 0
    try_num = 0
    ret = np.zeros(n)
    while cnt < n:
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
    return to_2pi_range(ret)


def cumsum_hist(
    mu: float, kappa: float, lambda_: float, bin_num: int
) -> npt.NDArray[np.float64]:
    """[0, 2pi] の間を bin_num (=D) 等分した区間でのcdfの値を返す。
    [F(i/D)] i=0,1,...,D を返す。
    """
    x = np.linspace(0, 2 * np.pi, bin_num + 1)
    y = sine_skewed_vonmises_periodic_cdf_analytical(x, mu, kappa, lambda_)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def W1_equal_div_cost_func(
    x, bin_num: int, data_cumsum_hist: npt.NDArray[np.float64]
) -> float:
    mu, kappa, lambda_ = x
    dist_cumsum_hist = cumsum_hist(mu, kappa, lambda_, bin_num)
    return method2.method2(data_cumsum_hist[1:], dist_cumsum_hist[1:])


def W1_equal_div(
    given_data: npt.NDArray[np.float64],
    bounds=((0, 2 * np.pi), (0.01, 10.0), (-1.0, 1.0)),
    tol=1e-7,
) -> optimize.OptimizeResult:
    """1-Wasserstein 距離（等分割ヒストグラム）を最小化するパラメータ推定"""
    given_data = to_2pi_range(given_data)
    bin_num = len(given_data)
    data_cumsum_hist = cumsum_hist_data(given_data, bin_num)
    cost_func = partial(
        W1_equal_div_cost_func, bin_num=bin_num, data_cumsum_hist=data_cumsum_hist
    )
    return optimize.differential_evolution(
        cost_func,
        tol=tol,
        bounds=bounds,
    )
