"""
連続分布のパラメータを、経験分布関数との2-ワッサースタイン距離最小化で推定します。
このとき、連続分布からサンプリングした離散分布で、元の連続分布を近似します。
"""

import matplotlib.pylab as plt
import numpy as np
import ot
from scipy.special import iv

def sample_from_von_mises(loc: float, kappa: float, N: int) -> np.ndarray:
    """フォンミーゼス分布からN点サンプルする。
    範囲は [0, 2pi]。
    """
    return np.random.vonmises(loc, kappa, size=N) + np.pi

def main():
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    n = 1000
    mu1 = 0.1
    kappa1 = 2
    mu2 = 0
    kappa2 = kappa1
    given_data = sample_from_von_mises(mu1, kappa1, n)
    ticks = [0, 0.15, 0.3]
    plt.figure()
    plt.subplot(projection="polar")
    plt.yticks(ticks)
    plt.hist(given_data, density=True, bins=int(np.sqrt(n)))
    plt.show()

if __name__ == "__main__":
    main()
