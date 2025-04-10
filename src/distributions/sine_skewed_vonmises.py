import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.stats import vonmises
from scipy.special import i0, i1, iv


def pdf(x: npt.NDArray[np.float64], mu: float, kappa: float, lambda_: float) -> npt.NDArray[np.float64]:
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

def rejection_sampling(n: int, mu: float, kappa: float, lambda_: float, debug: False) -> npt.NDArray[np.float64]:
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
        xs = stats.vonmises.rvs(2 * (n - cnt)) # 一気に 2 * 必要数サンプルする (棄却率が50%のため)
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


def _main():
    pass


if __name__ == "__main__":
    _main()