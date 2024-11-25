import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises

EPS = 1e-10

def calc_w_dist(sample, loc: float, kappa: float) -> float:
    """ `sample` の経験分布関数と、フォンミーゼス分布のpdfとの2-Wasserstein距離を計算する

    Args:
        sample : いくつかの円周上のデータ、-piからpiまでの範囲
        loc (float): 位置パラメータ
        kappa (float): 集中度パラメータ
    """
    return 1.0

def plot_vonmises_cdf(loc: float, kappa: float):
    """メモ: CDFはscipyの実装だと-piで0にならないのでずらしている
    scipyの実装では、locで0.5になるようになっている
    """
    dist = vonmises(loc=loc, kappa=kappa)
    x = np.linspace(-np.pi, np.pi, 1000)
    y = dist.cdf(x) - dist.cdf(-np.pi + EPS)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    x2 = np.linspace(0, 1, 100)
    y2 = np.remainder(dist.ppf(x2) + np.pi, 2*np.pi) - np.pi
    print(dist.cdf(loc))
    print(dist.ppf(0.5))

    fig2, ax2 = plt.subplots()
    ax2.plot(x2, y2)
    plt.show()

def main():
    loc = 0.5 * np.pi  # circular mean
    kappa = 1  # concentration
    N = 10000
    sample = vonmises(loc=loc, kappa=kappa).rvs(N)

    # another_dist_loc = 0
    # another_dist_kappa = 2

    # dist = calc_w_dist(sample, another_dist_loc, another_dist_kappa)

    # print(f"2-Wasserstein distance: {dist}")

    plot_vonmises_cdf(loc, kappa)

if __name__ == "__main__":
    main()