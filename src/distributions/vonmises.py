from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.special import i0, i1, iv
from scipy.stats import vonmises


def fisher_info_2x2(kappa: float) -> npt.NDArray[np.float64]:
    """
    フォンミーゼス分布のフィッシャー情報量を計算する
    """
    return np.array(
        [
            [kappa * i1(kappa) / i0(kappa), 0],
            [0, (1 + iv(2, kappa) / i0(kappa)) / 2 - (i1(kappa) / i0(kappa)) ** 2],
        ]
    )


def fisher_mat_inv_diag(kappa: float) -> List[float]:
    """フィッシャー情報行列の逆行列の対角成分のリストを返す。

    Returns:
        List[float]: [mu, kappa] の順
    """
    mat = fisher_info_2x2(kappa)  # 対角行列なので逆数が逆行列
    return [1 / mat[0][0], 1 / mat[1][1]]


def T(x):
    """フォンミーゼス分布の十分統計量を返す

    Args:
        x: フォンミーゼス分布からのサンプル。2pi周期。

    Returns:
        List[float, float]: 十分統計量、[cos, sin] の順
    """
    return [np.sum(np.cos(x)), np.sum(np.sin(x))]


def MLE(T_data, N: int):
    """最尤推定を行う

    Args:
        T_data: 十分統計量
        N(int): サンプル数

    Returns:
        Tuple[float, float]: 最尤推定値、[mu_MLE, kappa_MLE] の順
    """
    mu_MLE = np.arctan2(T_data[1], T_data[0])
    target_value = (T_data[0] * np.cos(mu_MLE) + T_data[1] * np.sin(mu_MLE)) / N
    # ここから二分探索による数値計算で逆関数を求める
    EPS = 1e-6
    left = EPS
    right = 1000  # これ以上大きくするとベッセル関数が発散(オーバーフロー)してしまう！
    while right - left > EPS:
        mid = (left + right) / 2
        now_value = i1(mid) / i0(mid)
        # print(mid, i0(mid), i1(mid), now_value) # for debug
        if np.abs(now_value - target_value) < EPS:
            break
        elif now_value - target_value > 0:
            right = mid
        else:
            left = mid
    kappa_MLE = mid
    return [mu_MLE, kappa_MLE]


def cumsum_hist(mu: float, kappa: float, bin_num: int) -> npt.NDArray[np.float64]:
    """[-pi, pi] の間を bin_num (=D) 等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    dist = vonmises(loc=mu, kappa=kappa)
    x = np.linspace(-np.pi, np.pi, bin_num + 1)
    y = dist.cdf(x) - dist.cdf(-np.pi)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7, print(f"{mu}, {kappa}, {y}")
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def cumsum_hist_data(data: npt.NDArray[np.float64], bin_num) -> npt.NDArray[np.float64]:
    """[-pi, pi] の間を bin_num (=D) 等分した区間でのデータのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    n = len(data)
    data_hist = np.zeros(bin_num + 1)
    for x in data:
        data_hist[
            np.clip(
                int(np.remainder(bin_num * (x + np.pi) / (2 * np.pi), bin_num)) + 1,
                1,
                bin_num,
            )
        ] += 1
    data_cumsum_hist = np.cumsum(data_hist) / n

    assert abs(data_cumsum_hist[0] - 0.0) < 1e-7
    assert abs(data_cumsum_hist[-1] - 1.0) < 1e-7
    return data_cumsum_hist


def quantile_sampling(
    mu: float, kappa: float, sample_num: int
) -> npt.NDArray[np.float64]:
    """フォンミーゼス分布から分位点サンプリングする

    Args:
        mu (float): 分布のパラメータ
        kappa (float): 分布のパラメータ
        sample_num (int): サンプルする数

    Returns:
        npt.NDArray[np.float64]: [-pi, pi] の範囲のサンプル。F^(-1)(i/D) (i=0, 1, ..., D)
    """
    eps = 1e-7
    x = np.linspace(
        eps, 1 - eps, sample_num
    )  # なぜか0, 1のppfを計算するとinfになるので避ける。
    dist = vonmises(loc=mu, kappa=kappa)
    y = dist.ppf(x)
    y2 = np.remainder(y + np.pi, 2 * np.pi) - np.pi
    assert np.all((-np.pi <= y2) & (y2 <= np.pi))
    return y2


def fast_quantile_sampling(
    mu: float, kappa: float, sample_num: int
) -> npt.NDArray[np.float64]:
    """フォンミーゼス分布から簡易的な分位点サンプリングする

    Args:
        mu (float): 分布のパラメータ
        kappa (float): 分布のパラメータ
        sample_num (int): サンプルする数

    Returns:
        npt.NDArray[np.float64]: [mu-pi, mu+pi] の範囲のサンプル。F^(-1)(i/D) (i=0, 1, ..., D)
    """
    x, step = np.linspace(0, 1, sample_num, endpoint=False, retstep=True)
    x = x + step / 2
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
    # now (lefts, lefts + step) に xi がある。中点で代表。
    return lefts + step / 2


def circular_variance(kappa: float) -> float:
    """フォンミーゼス分布の円周分散を計算する

    Args:
        kappa (float): 分布のパラメータ

    Returns:
        float: 円周分散
    """
    R = i1(kappa) / i0(kappa)
    return 1 - R


def A0(kappa: float) -> float:
    """A0関数を計算する。
    A0(0)=0, A0(inf)=1 の単調増加関数。
    大きな値だとオーバーフローするので注意。

    Args:
        kappa (float): 分布のパラメータ

    Returns:
        float: A0(kappa) の値
    """
    return i1(kappa) / i0(kappa)


def A0Inverse(y: float) -> float:
    """A0の逆関数を数値的に求める
    A0はA0(kappa) = I1(kappa) / I0(kappa) で定義される関数で、kappa >= 0 の単調増加関数。
    Ap(0)=0, Ap(inf)=1

    Args:
        y (float): A0(kappa) の値. 0 <= y < 1

    Returns:
        float: kappa の値
    """
    EPS = 1e-6
    left = EPS
    right = 1000  # これ以上大きくするとベッセル関数が発散(オーバーフロー)してしまう！
    while right - left > EPS:
        mid = (left + right) / 2
        now_value = i1(mid) / i0(mid)
        if np.abs(now_value - y) < EPS:
            break
        elif now_value - y > 0:
            right = mid
        else:
            left = mid
    return mid


def type0_estimate(data: npt.NDArray[np.float64], gamma: float, debug: bool=False) -> List[float]:
    """type0推定量を計算する
    Kato and Eguchi (2016)

    Args:
        data (npt.NDArray[np.float64]): サンプルデータ
        gamma (float): ハイパーパラメータ
        debug (bool): デバッグ用のフラグ。Trueのとき、更新ごとに推定値を出力する。

    Returns:
        List[float]: 推定値 [mu, kappa] の順
    """
    T_data = T(data)
    N = len(data)
    initial_guess = MLE(T_data, N)  # 最尤推定値を初期値として使用

    now_mu: float = initial_guess[0]
    now_kappa: float = initial_guess[1]

    for i in range(1000):  # 最大1000回の更新
        # 更新式
        next_mu = now_mu
        next_kappa = now_kappa

        w = np.exp(gamma * now_kappa * np.cos(data - now_mu))
        w_sum = np.sum(w)
        y = np.sum(w * np.sin(data))
        x = np.sum(w * np.cos(data))
        next_mu = np.arctan2(y, x)
        target = np.hypot(x, y) / w_sum
        next_kappa = A0Inverse(target) / (1 + gamma)

        if (next_mu - now_mu) ** 2 + (next_kappa - now_kappa) ** 2 < 1e-16:
            break   # 収束判定
        now_mu = next_mu
        now_kappa = next_kappa
        if debug:
            print(f"debug: i={i}, mu={now_mu}, kappa={now_kappa}")
    return [now_mu, now_kappa]


def type1_estimate(data: npt.NDArray[np.float64], beta: float, debug: bool=False) -> List[float]:
    """type1推定量を計算する
    Kato and Eguchi (2016)

    Args:
        data (npt.NDArray[np.float64]): サンプルデータ
        beta (float): ハイパーパラメータ
        debug (bool): デバッグ用のフラグ。Trueのとき、更新ごとに推定値を出力する。

    Returns:
        List[float]: 推定値 [mu, kappa] の順
    """
    T_data = T(data)
    N = len(data)
    initial_guess = MLE(T_data, N)  # 最尤推定値を初期値として使用

    now_mu: float = initial_guess[0]
    now_kappa: float = initial_guess[1]

    for i in range(1000):  # 最大1000回の更新
        # 更新式
        next_mu = now_mu
        next_kappa = now_kappa

        w = np.exp(beta * now_kappa * np.cos(data - now_mu))
        w_sum = np.sum(w)
        y = np.sum(w * np.sin(data))
        x = np.sum(w * np.cos(data))
        next_mu = np.arctan2(y, x)
        D = i0((1 + beta) * now_kappa) / i0(now_kappa) * (A0((1 + beta) * now_kappa) - A0(now_kappa)) / now_kappa
        target = np.hypot(x - N * D * np.cos(now_mu), y - N * D * np.sin(now_mu)) / w_sum
        next_kappa = A0Inverse(target)

        if (next_mu - now_mu) ** 2 + (next_kappa - now_kappa) ** 2 < 1e-16:
            break   # 収束判定
        now_mu = next_mu
        now_kappa = next_kappa
        if debug:
            print(f"debug: i={i}, mu={now_mu}, kappa={now_kappa}")
    return [now_mu, now_kappa]


def _plot_for_slide():
    """スライドに載せる分布の例の画像を作成する"""
    n = 100000
    mu = 0
    kappa = 2
    fig = plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")
    x = np.linspace(-np.pi, np.pi, 1000)
    vonmises_pdf = vonmises.pdf(x, loc=mu, kappa=kappa)
    sample = fast_quantile_sampling(mu, kappa, n)
    ticks = [0, 0.15, 0.3]

    left.plot(x, vonmises_pdf)
    left.set_yticks(ticks)
    number_of_bins = int(np.sqrt(n))
    left.hist(sample, density=True, bins=number_of_bins)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x, vonmises_pdf, label="PDF")
    right.set_yticks(ticks)
    right.hist(sample, density=True, bins=number_of_bins, label="Histogram")
    right.set_title("Polar plot")

    right.legend(bbox_to_anchor=(0.15, 1.06))
    plt.show()


def _estimate():
    """
    いろんな推定量を計算してみる
    """
    mu = 0.5 * np.pi + 2 * np.pi  # circular mean
    kappa = 1.3  # concentration
    N = 10000
    dist = vonmises(loc=mu, kappa=kappa)
    sample = dist.rvs(N)
    T_data = T(sample)
    mu_MLE, kappa_MLE = MLE(T_data, N)
    print(f"MLE: mu={mu_MLE}, kappa={kappa_MLE}")
    mu_type0, kappa_type0 = type0_estimate(sample, gamma=0, debug=True)
    print(f"type0 estimator: mu={mu_type0}, kappa={kappa_type0}")
    mu_type1, kappa_type1 = type1_estimate(sample, beta=0, debug=True)
    print(f"type1 estimator: mu={mu_type1}, kappa={kappa_type1}")
    mu_type0, kappa_type0 = type0_estimate(sample, gamma=0.5, debug=True)
    print(f"type0 estimator: mu={mu_type0}, kappa={kappa_type0}")
    mu_type1, kappa_type1 = type1_estimate(sample, beta=0.5, debug=True)
    print(f"type1 estimator: mu={mu_type1}, kappa={kappa_type1}")


def _main():
    mu = 0.5 * np.pi + 2 * np.pi  # circular mean
    kappa = 1.3  # concentration
    N = 10000
    dist = vonmises(loc=mu, kappa=kappa)

    # calc Fisher info matrix
    print("Fisher info:")
    print(fisher_info_2x2(kappa))

    sample = dist.rvs(N)
    print(f"min: {np.min(sample)}, max: {np.max(sample)}")  # [-pi, pi]

    sample2 = np.remainder(sample, 2 * np.pi)
    print(f"min: {np.min(sample2)}, max: {np.max(sample2)}")  # [0, 2pi]

    # calc MLE
    T_data = T(sample)
    mu_MLE, kappa_MLE = MLE(T_data, N)
    print(f"MLE: mu={mu_MLE}, kappa={kappa_MLE}")

    sample2 = quantile_sampling(mu, kappa, N)
    print(f"min: {np.min(sample2)}, max: {np.max(sample2)}")

    # plots
    # 普通のCDFは周期拡張されていて、平均で0.5になるようになっている。
    # 不便なので、-piで0、piで1になるようなCDFに変換する。
    print(dist.cdf(mu))  # 0.5
    x = np.linspace(-np.pi, np.pi, 1001)
    plt.plot(x, dist.pdf(x), label="pdf")
    plt.plot(x, dist.cdf(x), label="cdf")
    plt.plot(x, dist.cdf(x) - dist.cdf(-np.pi), label="normalized cdf")
    plt.legend()
    plt.show()

    # 普通のPPFは周期拡張されていて、0で平均-pi, 1で平均+piになるようになっている。
    # つまり、普通のCDFの逆関数になるようになっている。
    # 0で-pi、1でpiになるようなPPFにすることもできるが、多くの場合する必要はない。
    eps = 1e-7
    x = np.linspace(eps, 1 - eps, 1001)
    y = dist.ppf(x)
    plt.plot(x, y, label="ppf")
    # y2 = dist.ppf(x + dist.cdf(-np.pi))
    # plt.plot(x, y2, label="normalized ppf")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _estimate()
    # _main()
    # _plot_for_slide()
