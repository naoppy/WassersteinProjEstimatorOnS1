import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wrapcauchy
import numpy.typing as npt

"""
scipyの巻き込みコーシー分布の使い方メモ
cは分布の形状パラメータ
locは分布の平均値パラメータ
scaleはデフォルトの分布が [0, 2*pi] だが、scaleを使うと [0, 2*pi*scale] とかにできそう？
[0, 1] にするときに使えそう。今回は使わなくて良さそう。

さて、ここからが問題なのだが、wrapcauchyはvonmisesと違って周期性のないcdf, pdfの定義になっている
例えば pdf は [loc, loc+2*pi] で定義されていてそれ以外の範囲では 0 になる
cdf も同様で [loc, loc+2*pi] で定義されていてそれ以外の範囲では 0 になる
これを周期性があるように直す必要がある
"""


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


def main():
    mu = 0
    rho = 0.8
    dist = wrapcauchy(loc=mu, c=rho)
    x = np.linspace(-2 * np.pi, 4 * np.pi, 3000, endpoint=True)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(221)
    ax1.plot(x, dist.pdf(x))
    ax1.set_title("pdf")
    ax2 = fig.add_subplot(222)
    ax2.plot(x, wrapcauchy_true_pdf(x, rho, mu, 1))
    ax2.set_title("periodic pdf")
    ax3 = fig.add_subplot(223)
    ax3.plot(x, dist.cdf(x))
    ax3.set_title("cdf")
    ax4 = fig.add_subplot(224)
    ax4.plot(
        x,
        wrapcauchy_periodic_cdf(x, rho, mu, 1) - wrapcauchy_periodic_cdf(0, rho, mu, 1),
    )
    ax4.set_title("periodic cdf")
    plt.show()

    # y = dist.pdf(x)
    # plt.plot(x, y)
    # plt.show()
    # plt.subplot(projection="polar")
    # plt.plot(x, y)
    # plt.show()

    # y = wrapcauchy_true_pdf(x, rho, mu, 1)
    # plt.plot(x, y)
    # plt.show()

    # y = dist.cdf(x)
    # plt.plot(x, y)
    # plt.show()

    # y = wrapcauchy_periodic_cdf(x, rho, mu, 1)
    # plt.plot(x, y)
    # plt.show()


if __name__ == "__main__":
    main()
