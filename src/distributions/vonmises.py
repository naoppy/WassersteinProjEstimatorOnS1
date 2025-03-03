import numpy as np
import numpy.typing as npt
from scipy.special import i0, i1, iv
from scipy.stats import vonmises


def fisher_info_2x2(kappa: float) -> npt.NDArray[np.float64]:
    """
    フォンミーゼス分布のフィッシャー情報量を計算する
    """
    return np.array(
        [
            [kappa * i1(kappa) / i0(kappa), 0],
            [0, 1 / 2 + iv(2, kappa) / i0(kappa) - (i1(kappa) / i0(kappa)) ** 2],
        ]
    )


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


def cumsum_hist(mu: float, kappa: float, bin_num) -> npt.NDArray[np.float64]:
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
        npt.NDArray[np.float64]: [-pi, pi] の範囲のサンプル
    """
    x = np.linspace(-np.pi, np.pi, sample_num + 1)
    dist = vonmises(loc=mu, kappa=kappa)
    y = dist.ppf(x)
    assert np.all((-np.pi <= y) and (y <= np.pi))
    return y


def main():
    loc = 0.5 * np.pi  # circular mean
    kappa = 1  # concentration
    N = 10000
    sample = vonmises(loc=loc, kappa=kappa).rvs(N)

    # plot_vonmises(sample, loc, kappa, N)

    T_data = T(sample)
    mu_MLE, kappa_MLE = MLE(T_data, N)
    print(mu_MLE, kappa_MLE)


if __name__ == "__main__":
    main()
